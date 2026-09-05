import unittest
from scripts.audit_pf_internal_rule_ablation import rounded_pair, audit

class InternalRuleAuditTest(unittest.TestCase):
    def test_direct_rounding_avoids_double_rounding(self):
        self.assertEqual(rounded_pair(4.830562284968709, 0.245070096633694), '4.8 ± 0.2')
        self.assertEqual(rounded_pair(899.2, 14.292655456562313), '900 ± 10')

    def test_archived_s6_results(self):
        report = audit()
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['task_results'], 9000)
        self.assertEqual(report['unique_tasks'], 1500)
        self.assertEqual(report['exact_McNemar_Holm_tests'], 5)
        self.assertAlmostEqual(report['successful_additions_differences']['no_required_volume_term'], 0.256085915603348)

if __name__ == '__main__':
    unittest.main()
