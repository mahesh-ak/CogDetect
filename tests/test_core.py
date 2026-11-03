import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cogtran2 import align_msa_sca as package_align_msa_sca
from cogtran2 import upgma_clusters as package_upgma_clusters
from cogtran2.training import (
    DEFAULT_PREDICTION_THRESHOLD,
    _feature_for_concept,
    _create_optimizer_and_scheduler,
    _load_dataset,
    _make_config,
    _output_dir,
    _precompute_alignments,
    _run_epoch,
    _training_alignment_tokens,
    add_common_arguments,
)
from cogtran2.data import (
    align_msa_sca,
    asjp_symbols,
    bcubed,
    Concept,
    read_concepts,
    upgma_clusters,
)
from cogtran2.data_collator import CogTranDataCollator
from cogtran2.models import CogTranTriAttn, MSAConfig
from cogtran2.pretrained import load_cogtran_pretrained
from cogtran2.tokenizer import CogTranTokenizer


class AlignmentTests(unittest.TestCase):
    def test_pipeline_helpers_are_exported_from_package(self):
        self.assertIs(package_align_msa_sca, align_msa_sca)
        self.assertIs(package_upgma_clusters, upgma_clusters)

    def test_cogdetect_lingpy_alignment_and_sca_conversion(self):
        aligned = align_msa_sca((tuple("ɲata"), tuple("ɲta")))
        self.assertEqual(aligned, [["H", "a", "t", "a"], ["H", "~", "t", "a"]])

    def test_asjp_input_bypasses_ipa_conversion(self):
        with patch("cogtran2.data._asjp_converter", side_effect=AssertionError):
            aligned = align_msa_sca((tuple("Hata"), tuple("Hta")), asjp=True)
        self.assertEqual(aligned, [["H", "a", "t", "a"], ["H", "~", "t", "a"]])

    def test_asjp_click_symbols_are_removed_before_alignment(self):
        aligned = align_msa_sca((tuple("ǃa"), tuple("a")))
        self.assertNotIn("!", {symbol for form in aligned for symbol in form})
        self.assertEqual(len(aligned[0]), len(aligned[1]))
        self.assertEqual(aligned[0], aligned[1])

    def test_read_concepts_retains_raw_ipa_for_lingpy_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "D_test_Family-1.csv"
            data_file.write_text(
                "TOKENS,COGID,CONCEPT\nɲ a,1,hand\nɲ t a,1,hand\n",
                encoding="utf-8",
            )
            concepts = read_concepts(directory)
        self.assertEqual(concepts[0].forms, (("ɲ", "a"), ("ɲ", "t", "a")))
        self.assertEqual(concepts[0].languages, ("", ""))

    def test_read_concepts_retains_doculects_for_language_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "D_test_Family-1.csv"
            data_file.write_text(
                "DOCULECT\tTOKENS\tCOGID\tCONCEPT\nLangA\tɲ a\t1\thand\nLangB\tɲ t a\t1\thand\n",
                encoding="utf-8",
            )
            concepts = read_concepts(directory)
        self.assertEqual(concepts[0].languages, ("LangA", "LangB"))

    def test_bcubed_and_upgma(self):
        similarities = [[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]]
        clusters = upgma_clusters(similarities, 0.5)
        self.assertEqual(clusters[0], clusters[1])
        self.assertNotEqual(clusters[0], clusters[2])
        self.assertEqual(bcubed(clusters, [4, 4, 7]), (1.0, 1.0, 1.0))

    def test_upgma_matches_lingpy_on_symmetrized_similarity(self):
        similarities = [
            [1.0, 0.8, 0.2],
            [0.3, 1.0, 0.1],
            [0.9, 0.1, 1.0],
        ]
        symmetric_distances = [
            [1.0 - (similarities[i][j] + similarities[j][i]) / 2 for j in range(3)]
            for i in range(3)
        ]
        from lingpy.algorithm.clustering import flat_cluster

        reference = [0] * 3
        for label, members in enumerate(
            flat_cluster("upgma", 0.6, symmetric_distances).values()
        ):
            for member in members:
                reference[member] = label

        actual = upgma_clusters(similarities, threshold=0.4)
        self.assertEqual(
            [actual[i] == actual[j] for i in range(3) for j in range(3)],
            [reference[i] == reference[j] for i in range(3) for j in range(3)],
        )
        self.assertEqual(actual, [0, 0, 1])


class DatasetLayoutTests(unittest.TestCase):
    def test_training_defaults_and_distinct_run_directories(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.epochs, 20)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.msa_workers, 1)
        self.assertEqual(args.threshold, DEFAULT_PREDICTION_THRESHOLD)
        self.assertEqual(args.threshold, 0.4)
        config = _make_config(args, CogTranTokenizer(["A", "B"]))
        self.assertEqual(config.vocab_size, 768)
        self.assertEqual(
            (
                config.hidden_size,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.intermediate_size,
            ),
            (128, 2, 2, 128),
        )
        self.assertEqual(
            _output_dir(args),
            Path("runs/triattn/prop_125/split_0/no_language_tokens"),
        )
        args.split = 2
        self.assertEqual(
            _output_dir(args),
            Path("runs/triattn/prop_125/split_2/no_language_tokens"),
        )
        args.language_tokens = True
        self.assertEqual(_output_dir(args), Path("runs/triattn/prop_125/split_2/language_tokens"))
        self.assertTrue(_make_config(args, CogTranTokenizer(["A", "B"])).use_language_tokens)
        args.train_proportion = "0"
        self.assertEqual(_output_dir(args), Path("runs/triattn/prop_0/language_tokens"))
        args.language_tokens = False
        self.assertEqual(_output_dir(args), Path("runs/triattn/prop_0/no_language_tokens"))

    def test_default_split_uses_prop_125_directory(self):
        args = Namespace(
            data_dir=Path("data"),
            train_proportion="125",
            split=3,
            max_concepts=None,
        )
        with patch("cogtran2.training.read_concepts", return_value=[]) as read:
            _load_dataset(args)
        self.assertEqual(
            [call.args[0] for call in read.call_args_list],
            [
                Path("data/train"),
                Path("data/prop_125/split_3/train"),
                Path("data/prop_125/split_3/test"),
            ],
        )

    def test_prop_90_split_uses_generated_fold(self):
        args = Namespace(
            data_dir=Path("data"),
            train_proportion="90",
            split=0,
            max_concepts=None,
        )
        with patch("cogtran2.training.read_concepts", return_value=[]) as read:
            _load_dataset(args)
        self.assertEqual(
            [call.args[0] for call in read.call_args_list],
            [
                Path("data/train"),
                Path("data/prop_90/split_0/train"),
                Path("data/prop_90/split_0/test"),
            ],
        )

    def test_zero_proportion_uses_only_base_train_and_both_test_sets(self):
        args = Namespace(
            data_dir=Path("data"),
            train_proportion="0",
            split=0,
            max_concepts=None,
        )
        with patch("cogtran2.training.read_concepts", return_value=[]) as read:
            _load_dataset(args)
        self.assertEqual(
            [call.args[0] for call in read.call_args_list],
            [Path("data/train"), Path("data/test1"), Path("data/test2")],
        )


class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        self.config = MSAConfig(
            vocab_size=10,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_rows=8,
            max_columns=16,
        )

    def test_tokenizer_save_load_and_hf_encoding(self):
        tokenizer = CogTranTokenizer(["A", "B"])
        encoded = tokenizer("A B", return_tensors="pt")
        self.assertEqual(encoded["input_ids"].tolist(), [[0, 7, 8, 1]])
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            reloaded = CogTranTokenizer.from_pretrained(directory)
        self.assertEqual(reloaded.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded("A B")["input_ids"], [0, 7, 8, 1])

    def test_default_model_config_matches_cogdetect_triattn_dimensions(self):
        config = MSAConfig()
        self.assertEqual(
            (
                config.hidden_size,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.intermediate_size,
            ),
            (128, 2, 2, 128),
        )
        self.assertEqual(config.vocab_size, 768)

    def test_msa_layer_norms_match_fair_esm_epsilon(self):
        model = CogTranTriAttn(self.config)
        self.assertEqual(model.msa.emb_layer_norm_before.eps, 1e-5)
        self.assertEqual(model.msa.emb_layer_norm_after.eps, 1e-5)
        for layer in model.msa.layers:
            self.assertEqual(layer.row_self_attention.layer_norm.eps, 1e-5)
            self.assertEqual(layer.column_self_attention.layer_norm.eps, 1e-5)
            self.assertEqual(layer.feed_forward_layer.layer_norm.eps, 1e-5)
        self.assertEqual(model.outer_norm.eps, 1e-5)
        self.assertEqual(model.pair_norm.eps, self.config.layer_norm_eps)
        self.assertEqual(model.pairwise_norm.eps, self.config.layer_norm_eps)

    def test_default_tokenizer_uses_fixed_lingpy_asjp_inventory(self):
        tokenizer = CogTranTokenizer()
        self.assertEqual(set(tokenizer.tokens[:-1]), set(asjp_symbols()))
        self.assertEqual(tokenizer.tokens[-1], "~")
        self.assertEqual(tokenizer.vocab_size, len(asjp_symbols()) + 1 + 7)

    def test_training_token_inventory_includes_observed_composite_segments_only(self):
        concept = Concept(
            family="Fam",
            concept="hand",
            forms=(("a",), ("b",)),
            cognate_ids=("1", "1"),
        )
        with patch(
            "cogtran2.training.align_msa_sca",
            return_value=[["a", "ae"], ["a", "~"]],
        ):
            tokens = _training_alignment_tokens([concept])
        tokenizer = CogTranTokenizer(tokens)
        self.assertEqual(tokens, {"a", "ae", "~"})
        self.assertEqual(tokenizer.convert_tokens_to_ids("ae"), tokenizer.token_to_id["ae"])
        self.assertNotIn("z", tokenizer.token_to_id)

    def test_parallel_msa_alignment_matches_serial(self):
        concepts = [
            Concept(
                family="Fam",
                concept=f"concept-{index}",
                forms=forms,
                cognate_ids=("1", "1"),
            )
            for index, forms in enumerate(((tuple("ɲata"), tuple("ɲta")), (tuple("pala"), tuple("para"))))
        ]
        parallel = _precompute_alignments(concepts, workers=2)
        for concept in concepts:
            self.assertEqual(parallel[concept.forms], align_msa_sca(concept.forms))

    def test_language_tokens_match_cogdetect_prefix_and_are_encoded(self):
        language_tokens = ("[Fam;LangA]", "[Fam;LangB]")
        tokenizer = CogTranTokenizer((*asjp_symbols(), *language_tokens))
        concept = Concept(
            family="Fam",
            concept="hand",
            forms=(("ɲ", "a"), ("ɲ", "t", "a")),
            cognate_ids=("1", "1"),
            languages=("LangA", "LangB"),
        )
        feature = _feature_for_concept(concept, tokenizer, 32, use_language_tokens=True)
        self.assertEqual(
            [tokenizer.convert_ids_to_tokens(row[1]) for row in feature["input_ids"]],
            ["[Fam;LangA]", "[Fam;LangB]"],
        )
        self.assertTrue(all(row[1] != tokenizer.unk_token_id for row in feature["input_ids"]))

    def test_language_tokens_require_doculect_values(self):
        concept = Concept(
            family="Fam",
            concept="hand",
            forms=(("a",), ("b",)),
            cognate_ids=("1", "1"),
        )
        with self.assertRaisesRegex(ValueError, "DOCULECT"):
            _feature_for_concept(concept, CogTranTokenizer(), 32, use_language_tokens=True)

    def test_hf_data_collator_pads_msa_rows_and_pair_labels(self):
        tokenizer = CogTranTokenizer(["A", "B"])
        collator = CogTranDataCollator(tokenizer)
        batch = collator(
            [
                {"input_ids": [[0, 7, 1], [0, 8, 1]], "labels": [[1, 0], [0, 1]]},
                {"input_ids": [[0, 7, 1]], "labels": [[1]]},
            ]
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 2, 3))
        self.assertEqual(tuple(batch["attention_mask"].shape), (2, 2, 3))
        self.assertEqual(batch["labels"][1, 1, 1].item(), -100)

    def test_native_axial_model_backprop_and_hf_save_load(self):
        tokenizer = CogTranTokenizer(["A", "B"])
        model = CogTranTriAttn(MSAConfig(**{**self.config.to_dict(), "vocab_size": len(tokenizer)}))
        batch = CogTranDataCollator(tokenizer)(
            [
                {
                    "input_ids": [[0, 7, 1], [0, 8, 1]],
                    "labels": [[1, 0], [0, 1]],
                }
            ]
        )
        output = model(**batch)
        output.loss.backward()
        self.assertEqual(tuple(output.logits.shape), (1, 2, 2, 2))
        self.assertFalse(torch.allclose(output.logits, output.logits.transpose(1, 2)))
        model.eval()
        expected = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = CogTranTriAttn.from_pretrained(directory)
        reloaded_output = reloaded(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        self.assertTrue(torch.allclose(expected, reloaded_output.logits))

    def test_pretrained_encoder_remaps_tokens(self):
        source = CogTranTriAttn(self.config)
        source.msa.embed_tokens.weight.data[7].fill_(7.0)
        source.msa.embed_tokens.weight.data[8].fill_(8.0)
        tokenizer = CogTranTokenizer(["A", "B", "C"])
        target = CogTranTriAttn(MSAConfig(**{**self.config.to_dict(), "vocab_size": len(tokenizer)}))
        target.msa.embed_tokens.weight.data.zero_()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            torch.save(
                {f"msat.{key}": value for key, value in source.msa.state_dict().items()},
                checkpoint / "pytorch_model.bin",
            )
            (checkpoint / "tokenizer_config.json").write_text(
                json.dumps({"chars": ["B", "A"]}), encoding="utf-8"
            )
            load_cogtran_pretrained(target, checkpoint, tokenizer.token_to_id)
        self.assertTrue(torch.all(target.msa.embed_tokens.weight[7] == 8.0))
        self.assertTrue(torch.all(target.msa.embed_tokens.weight[8] == 7.0))

    def test_pairwise_attention_uses_cogdetect_head_width(self):
        config = MSAConfig(
            vocab_size=10,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=12,
            max_rows=8,
            max_columns=16,
        )
        model = CogTranTriAttn(config)
        layer = model.triangle_layers[0]
        self.assertEqual(layer.triangle_attention_outgoing.attn.dim_head, 12)
        self.assertTrue(hasattr(layer, "triangle_multiply_outgoing"))
        self.assertTrue(hasattr(layer, "triangle_multiply_ingoing"))
        self.assertTrue(layer.triangle_attention_ingoing.global_query_attn)


class TrainingRecipeTests(unittest.TestCase):
    def test_linear_scheduler_uses_zero_weight_decay(self):
        model = torch.nn.Linear(1, 1)
        optimizer, scheduler = _create_optimizer_and_scheduler(model, 1e-3, 4)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)
        for expected in (0.75e-3, 0.5e-3, 0.25e-3, 0.0):
            optimizer.step()
            scheduler.step()
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)

    def test_training_clips_gradient_norm_to_one(self):
        class LossModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(10.0))

            def forward(self, input_ids, attention_mask, labels):
                return SimpleNamespace(loss=self.weight.square())

        model = LossModel()
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0)
        batch = {
            "input_ids": torch.ones((1, 1, 1), dtype=torch.long),
            "attention_mask": torch.ones((1, 1, 1), dtype=torch.long),
            "labels": torch.ones((1, 1, 1), dtype=torch.long),
        }
        with patch(
            "torch.nn.utils.clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as clip_grad_norm:
            _run_epoch(model, [batch], optimizer, torch.device("cpu"))
        clip_grad_norm.assert_called_once()
        self.assertEqual(clip_grad_norm.call_args.kwargs["max_norm"], 1.0)




if __name__ == "__main__":
    unittest.main()
