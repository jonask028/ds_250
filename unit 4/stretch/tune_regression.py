"""Reproducible parcel-separated regression experiments for Unit 4.

Run with ../../.venv/bin/python tune_regression.py from this directory.
The existing control parcels are excluded from all parameter selection.
"""
from pathlib import Path
import json
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent


def load_data():
    source = pl.read_csv(HERE.parent / 'dwellings_ml.csv')
    neighborhoods = pl.read_csv(HERE.parent / 'dwellings_neighborhoods_ml.csv')
    unique = neighborhoods.unique(subset=['parcel'])
    # Fail rather than silently choose between conflicting neighborhood records.
    assert unique.height == neighborhoods.unique().height
    joined = source.join(unique, on='parcel', how='left', validate='m:1')
    return joined, neighborhoods


def features(data, version='base'):
    result = data.select(cs.numeric().exclude(['yrbuilt', 'before1980', 'syear', 'smonth']))
    if version == 'engineered':
        result = result.with_columns(
            (pl.col('livearea') / pl.col('stories').clip(lower_bound=1)).alias('area_per_story'),
            (pl.col('livearea') / pl.col('numbdrm').clip(lower_bound=1)).alias('area_per_bedroom'),
            (pl.col('numbaths') / pl.col('numbdrm').clip(lower_bound=1)).alias('baths_per_bedroom'),
            (pl.col('sprice') / pl.col('livearea').clip(lower_bound=1)).alias('price_per_area'),
            (pl.col('basement') / pl.col('livearea').clip(lower_bound=1)).alias('basement_per_area'),
            (pl.col('finbsmnt') / pl.col('basement').clip(lower_bound=1)).alias('finished_basement_ratio'),
        )
    return result.to_numpy().astype(np.float32)


def make_model(spec):
    kind = spec['family']
    cls = {'extra': ExtraTreesRegressor, 'forest': RandomForestRegressor,
           'hist': HistGradientBoostingRegressor}[kind]
    params = dict(spec.get('params', {}))
    params['random_state'] = 28
    if kind != 'hist':
        params.setdefault('n_estimators', 150)
        params['n_jobs'] = -1
    return cls(**params)


def window_prediction(tree_predictions, radius):
    """Choose a dense interval of tree predictions; tune only on validation data."""
    result = np.empty(tree_predictions.shape[1])
    for i, values in enumerate(tree_predictions.T):
        ordered = np.sort(values)
        ends = np.searchsorted(ordered, ordered + 2 * radius, side='right')
        counts = ends - np.arange(len(ordered))
        candidates = np.flatnonzero(counts == counts.max())
        median = np.median(ordered)
        centers = (ordered[candidates] + ordered[ends[candidates] - 1]) / 2
        start = candidates[np.argmin(abs(centers - median))]
        end = ends[start]
        # This interval center covers the selected predictions within the radius.
        result[i] = np.clip(np.median(ordered[start:end]), ordered[end - 1] - radius, ordered[start] + radius)
    return result


def prediction_options(model, x):
    mean = model.predict(x)
    yield 'mean', mean
    if not hasattr(model, 'estimators_') or isinstance(model, HistGradientBoostingRegressor):
        return
    trees = np.asarray([tree.predict(x) for tree in model.estimators_])
    median = np.median(trees, axis=0)
    yield 'median', median
    yield 'mean_median_half', (mean + median) / 2
    yield 'window5', window_prediction(trees, 5)
    yield 'window10', window_prediction(trees, 10)


def predict_selected(model, x, aggregation):
    if aggregation == 'mean':
        return model.predict(x)
    trees = np.asarray([tree.predict(x) for tree in model.estimators_])
    if aggregation == 'median':
        return np.median(trees, axis=0)
    if aggregation == 'mean_median_half':
        return (model.predict(x) + np.median(trees, axis=0)) / 2
    return window_prediction(trees, int(aggregation.replace('window', '')))


def metrics(actual, predicted):
    absolute = abs(predicted - actual)
    old = actual < 1943
    return {'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSE': float(root_mean_squared_error(actual, predicted)),
            'R_squared': float(r2_score(actual, predicted)),
            'percent_over_5': float(100 * (absolute > 5).mean()),
            'percent_over_10': float(100 * (absolute > 10).mean()),
            'pre1943_MAE': float(absolute[old].mean()),
            'pre1943_percent_over_10': float(100 * (absolute[old] > 10).mean())}


def candidate_specs():
    specs = []
    for family in ['extra', 'forest']:
        for maximum in [1.0, .5, 'sqrt']:
            specs.append({'name': f'{family}_features_{maximum}', 'family': family,
                          'params': {'max_features': maximum}, 'features': 'base', 'training': 'corrected'})
        specs.extend([
            {'name': f'{family}_leaf2', 'family': family, 'params': {'min_samples_leaf': 2}, 'features': 'base', 'training': 'corrected'},
            {'name': f'{family}_engineered', 'family': family, 'params': {}, 'features': 'engineered', 'training': 'corrected'},
            {'name': f'{family}_original_join', 'family': family, 'params': {}, 'features': 'base', 'training': 'original'},
        ])
    specs.extend([
        {'name': 'extra_bootstrap', 'family': 'extra', 'params': {'bootstrap': True}, 'features': 'base', 'training': 'corrected'},
        {'name': 'forest_no_bootstrap', 'family': 'forest', 'params': {'bootstrap': False, 'max_features': .75}, 'features': 'base', 'training': 'corrected'},
    ])
    for leaves in [15, 31, 63]:
        for loss in ['squared_error', 'absolute_error']:
            specs.append({'name': f'hist_{leaves}_{loss}', 'family': 'hist',
                          'params': {'max_iter': 400, 'learning_rate': .07, 'max_leaf_nodes': leaves, 'min_samples_leaf': 10, 'l2_regularization': 1., 'loss': loss},
                          'features': 'engineered', 'training': 'corrected'})
    return specs


def training_rows(data, neighborhoods, parcels, mode):
    rows = data.filter(pl.col('parcel').is_in(parcels))
    if mode == 'original':
        # Reproduce the original join's weighting without altering evaluation rows.
        rows = rows.join(neighborhoods.select('parcel'), on='parcel', how='inner')
    return rows


def evaluate_spec(data, neighborhoods, development, spec, seed):
    training, validation = train_test_split(development, test_size=.20, random_state=seed)
    train = training_rows(data, neighborhoods, training, spec['training'])
    valid = data.filter(pl.col('parcel').is_in(validation))
    assert not set(train['parcel']) & set(valid['parcel'])
    model = make_model(spec)
    model.fit(features(train, spec['features']), train['yrbuilt'].to_numpy())
    records = []
    for agg, predicted in prediction_options(model, features(valid, spec['features'])):
        records.append({'name': spec['name'], 'aggregation': agg, 'seed': seed,
                        **metrics(valid['yrbuilt'].to_numpy(), predicted)})
    return records


def rank_key(row):
    return row['percent_over_10'], row['percent_over_5'], row['MAE'], row['name'], row['aggregation']


def main():
    data, neighborhoods = load_data()
    split = json.loads((HERE / 'unit4_stretch_control_split.json').read_text())
    development = split['training_parcels']
    control_parcels = split['control_parcels']
    assert not set(development) & set(control_parcels)
    specs = candidate_specs()
    records = []
    for spec in specs:
        rows = evaluate_spec(data, neighborhoods, development, spec, 7101)
        records.extend(rows)
        print('SEARCH', json.dumps(min(rows, key=rank_key)), flush=True)
        (HERE / 'unit4_stretch_tuning_validation.json').write_text(json.dumps(records, indent=2) + '\n')
    first = sorted(records, key=rank_key)
    names = list(dict.fromkeys(row['name'] for row in first))[:5]
    # Always retain defaults for an identical baseline comparison.
    for name in ['extra_features_1.0', 'forest_features_1.0']:
        if name not in names:
            names.append(name)
    by_name = {s['name']: s for s in specs}
    for seed in [7102, 7103]:
        for name in names:
            rows = evaluate_spec(data, neighborhoods, development, by_name[name], seed)
            records.extend(rows)
            print('CONFIRM', seed, json.dumps(min(rows, key=rank_key)), flush=True)
            (HERE / 'unit4_stretch_tuning_validation.json').write_text(json.dumps(records, indent=2) + '\n')
    averaged = []
    for name in names:
        aggregations = {r['aggregation'] for r in records if r['name'] == name}
        for agg in aggregations:
            matches = [r for r in records if r['name'] == name and r['aggregation'] == agg]
            assert len(matches) == 3
            averaged.append({'name': name, 'aggregation': agg,
                             **{k: float(np.mean([m[k] for m in matches])) for k in metrics(np.array([1900,2000]),np.array([1900,2000]))}})
    averaged.sort(key=rank_key)
    winner = averaged[0]
    configuration = {'spec': by_name[winner['name']], 'aggregation': winner['aggregation'],
                     'selection_priority': ['percent_over_10', 'percent_over_5', 'MAE'],
                     'validation': winner}
    configuration['alternatives'] = []
    for objective, label in [('MAE', 'Lowest validation MAE'),
                             ('percent_over_5', 'Fewest five-year misses')]:
        alternative = min(averaged, key=lambda row: (row[objective], *rank_key(row)))
        configuration['alternatives'].append({
            'model': label, 'spec': by_name[alternative['name']],
            'aggregation': alternative['aggregation'], 'validation': alternative
        })
    (HERE / 'unit4_stretch_best_model.json').write_text(json.dumps(configuration, indent=2) + '\n')
    (HERE / 'unit4_stretch_tuning_summary.json').write_text(json.dumps(averaged, indent=2) + '\n')
    print('WINNER', json.dumps(configuration), flush=True)
    # One final comparison after selection; do not pick settings based on this result.
    evaluation = data.filter(pl.col('parcel').is_in(control_parcels))
    final = []
    for label, spec, agg in [
        ('Default Extra Trees', by_name['extra_features_1.0'], 'mean'),
        ('Default random forest', by_name['forest_features_1.0'], 'mean'),
        ('Fewest ten-year misses', configuration['spec'], configuration['aggregation']),
        *[(a['model'], a['spec'], a['aggregation']) for a in configuration['alternatives']]
    ]:
        train = training_rows(data, neighborhoods, development, spec['training'])
        model = make_model(spec)
        model.fit(features(train,spec['features']),train['yrbuilt'].to_numpy())
        predicted = predict_selected(model, features(evaluation,spec['features']),agg)
        row = {'model':label, **metrics(evaluation['yrbuilt'].to_numpy(),predicted)}
        final.append(row)
        print('CONTROL', json.dumps(row), flush=True)
    report = {'training_parcels': len(development), 'control_parcels': len(control_parcels),
              'control_rows': evaluation.height, 'results': final,
              'note': 'Control parcels were excluded from this sweep, but have been evaluated earlier in the project. This is not a wholly untouched external test.'}
    (HERE / 'unit4_stretch_tuning_control.json').write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()
