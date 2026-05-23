import unittest

from ontomcda.ontology import load_profiles
from ontomcda.recommender import recommend_methods
from ontomcda.text_inference import infer_premises


class OntoMCDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_profiles("data/OntoMCDA_v2.owl")

    def test_loads_method_profiles(self):
        self.assertGreater(len(self.profiles), 50)
        first = next(iter(self.profiles.values()))
        self.assertIn("tipo_problema", first)

    def test_recommends_from_text(self):
        result = recommend_methods(
            "Preciso priorizar alternativas em ranking com criterios quantitativos e qualitativos, pesos e incerteza.",
            self.profiles,
            top_k=10,
        )

        self.assertEqual(result.premises["tipo_problema"], "Ordenacao")
        self.assertFalse(result.accepted.empty)

    def test_nlp_metrics_are_calculated(self):
        result = recommend_methods(
            "Desejo ordenar alternativas em grupo, com dados mistos, incerteza, pesos e criterios compensatorios.",
            self.profiles,
            top_k=5,
        )

        self.assertEqual(result.nlp_metrics["total_premises"], 10)
        self.assertGreater(result.nlp_metrics["inferred_premises"], 0)
        self.assertGreaterEqual(result.nlp_metrics["ics_pln"], 0)
        self.assertLessEqual(result.nlp_metrics["ics_pln"], 100)
        self.assertEqual(
            result.nlp_metrics["not_inferred_premises"],
            result.nlp_metrics["total_premises"] - result.nlp_metrics["inferred_premises"],
        )
        self.assertGreaterEqual(result.nlp_metrics["tni_pln"], 0)
        self.assertLessEqual(result.nlp_metrics["tni_pln"], 100)
        self.assertGreaterEqual(result.nlp_metrics["iet_pln"], 0)
        self.assertLessEqual(result.nlp_metrics["iet_pln"], 100)
        self.assertGreaterEqual(result.nlp_metrics["icl_pln"], 0)
        self.assertLessEqual(result.nlp_metrics["icl_pln"], 100)
        self.assertGreaterEqual(result.nlp_metrics["ijl_pln"], 0)
        self.assertLessEqual(result.nlp_metrics["ijl_pln"], 100)
        self.assertEqual(len(result.premise_diagnostics), 10)
        self.assertIn("Diagnostico", result.premise_diagnostics[0])

    def test_compensatoriedade_semantic_patterns(self):
        cases = {
            "um criterio pode compensar outro criterio": "Compensatorio",
            "compensação": "Compensatorio",
            "nao admite compensacao entre criterios": "NaoCompensatorio",
            "não compensatório": "NaoCompensatorio",
            "não compesnsatório": "NaoCompensatorio",
            "compensação parcial": "ParcialmenteCompensatorio",
            "a compensacao e parcial entre criterios": "ParcialmenteCompensatorio",
        }
        for text, expected in cases.items():
            premises, _, _ = infer_premises(text)
            self.assertEqual(premises["compensatoriedade"], expected)

    def test_difficult_premise_semantic_rules(self):
        cases = [
            ("os criterios seguem a regra quanto maior melhor", "monotonicidade", "Monotonico"),
            ("o aumento do valor nao implica melhora na preferencia", "monotonicidade", "NaoMonotonico"),
            ("os pesos definidos por comparacao par-a-par expressam importancia relativa", "ponderabilidade", "Ponderavel"),
            ("as preferencias incompletas geram comparacao parcial entre alternativas", "completude_pref", "Incompleto"),
            ("todas as alternativas possuem comparacoes completas", "completude_pref", "Completo"),
        ]
        for text, attr, expected in cases:
            premises, evidence, score_map = infer_premises(text)
            self.assertEqual(premises[attr], expected)
            self.assertGreaterEqual(score_map[attr], 2)
            self.assertTrue(evidence[attr])

    def test_expanded_lexicon_realistic_decision_text(self):
        text = (
            "O comite decisor precisa ranquear fornecedores com criterios numericos e julgamentos qualitativos. "
            "Ha probabilidades desconhecidas e ambiente incerto. O modelo admite compensacao com veto, pois existe "
            "requisito minimo obrigatorio. Os criterios usam vetor de pesos, com pesos como entrada, e a preferencia "
            "tem faixa ideal para estoque, caracterizando comportamento nao monotonico. Todas as comparacoes foram realizadas."
        )

        premises, evidence, score_map = infer_premises(text)

        expected = {
            "tipo_problema": "Ordenacao",
            "estrutura_decisoria": "Grupo",
            "tipo_variavel": "Mista",
            "ambiente_decisao": "Incerteza",
            "compensatoriedade": "ParcialmenteCompensatorio",
            "ponderabilidade": "Ponderavel",
            "usa_pesos": "Sim",
            "requer_pesos": "Sim",
            "monotonicidade": "NaoMonotonico",
            "completude_pref": "Completo",
        }
        for attr, value in expected.items():
            self.assertEqual(premises[attr], value)
            self.assertGreaterEqual(score_map[attr], 2)
            self.assertTrue(evidence[attr])


if __name__ == "__main__":
    unittest.main()
