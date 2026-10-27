import pytest
from mavt.grain_prior.config import GrainPriorConfig, config_from_dict
from mavt.grain_prior.protocol import validate_manifest, pilot_config, BASELINE_REVISION


def test_pilot_preserves_gp3_payload_and_explicit_versions():
    cfg = pilot_config('shared_field_v1')
    cfg.check()
    assert (cfg.stratum, cfg.d_f, cfg.n_grains) == (1, 12, 256)
    assert cfg.state_numbers([]) == 6180
    assert cfg.state_numbers([52]*3) == 12242
    assert cfg.query_version == 1 and cfg.innovation_semantics == 'disabled'
    assert cfg.render_samples == 32 and cfg.w_lpips == 0
    assert config_from_dict({}).readout_version == 'legacy'
    assert len(BASELINE_REVISION) == 40
    with pytest.raises(ValueError):
        GrainPriorConfig(readout_version='typo').check()


def test_manifest_requires_disjoint_groups_and_source_target_ids():
    rows = [dict(scene_id='s', group_id='g', split='val', source_ids=['a'], target_ids=['b'])]
    assert validate_manifest(rows)['scenes'] == 1
    for bad in [rows + [dict(rows[0], split='train')],
                [dict(rows[0], target_ids=['a'])], [dict(rows[0], group_id='')]]:
        with pytest.raises(ValueError):
            validate_manifest(bad)
