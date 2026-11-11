import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import analyze,demo_events
from app import create_app

def config(dataset='demo_uplift'):
    return dict(name='test',dataset_id=dataset,start='2026-01-01Z'.replace('01Z','01T00:00:00Z'),end='2026-02-01T00:00:00Z',primary_metric='completion_rate',alpha=.05,mde=.05,allocation_b=.5,planned_segments=['new'])
class AnalysisTests(unittest.TestCase):
    def test_effect_and_dedup(self):
        r=analyze(demo_events('uplift'),config())
        self.assertEqual(sum(r['counts'].values()),1600)
        self.assertEqual(r['raw_exposure_events'],3200)
        self.assertTrue(r['metrics'][0]['significant']);self.assertGreater(r['metrics'][0]['effect'],0)
    def test_ratio_and_early_warning(self):
        r=analyze(demo_events('ratio_anomaly'),config(),now='2026-01-10T00:00:00Z')
        self.assertLess(r['srm_p_value'],.001);self.assertFalse(r['finished'])
    def test_versioned_api(self):
        with tempfile.TemporaryDirectory() as d:
            c=create_app(d).test_client();self.assertEqual(c.get('/').status_code,200)
            e=c.post('/api/experiments',json=config()).get_json();self.assertTrue(e['locked'])
            r=c.post('/api/analyze/'+e['id'],json={});self.assertEqual(r.status_code,200)
            self.assertEqual(c.get('/api/export/'+r.json['id']).status_code,200)
            self.assertEqual(c.put('/api/experiments/'+e['id'],json={}).status_code,405)
if __name__=='__main__':unittest.main()
