import unittest
from types import SimpleNamespace

from training.agenttune.sft_smoke import chat_token_length, normalize_trainable_precision, prepare_artifact_repo


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def float(self):
        return Tensor('torch.float32')


class Parameter:
    def __init__(self, dtype, trainable=True):
        self.data = Tensor(dtype)
        self.requires_grad = trainable
        self.grad = None

    @property
    def dtype(self):
        return self.data.dtype

    def is_floating_point(self):
        return True

    def numel(self):
        return 8


class SFTPrecisionTests(unittest.TestCase):
    def test_context_length_counts_tokens_in_encoding_not_mapping_fields(self):
        def render(messages, **kwargs):
            self.assertTrue(kwargs['return_dict'])
            return {'input_ids': list(range(4100)), 'attention_mask': [1]*4100}
        tokenizer = SimpleNamespace(apply_chat_template=render)
        self.assertEqual(chat_token_length(tokenizer, []), 4100)

    def test_trainable_bf16_and_fp16_adapters_become_fp32_frozen_base_unchanged(self):
        base = Parameter('torch.uint8', trainable=False)
        adapters = [Parameter('torch.bfloat16'), Parameter('torch.float16')]
        model = SimpleNamespace(named_parameters=lambda: iter([
            ('base.weight', base), ('layer.lora_A.weight', adapters[0]),
            ('layer.lora_B.weight', adapters[1])]))
        report = normalize_trainable_precision(model)
        self.assertEqual(base.dtype, 'torch.uint8')
        self.assertTrue(all(p.dtype == 'torch.float32' for p in adapters))
        self.assertEqual(report['after'], {'torch.float32': 16})

    def test_existing_gradients_and_non_adapter_trainables_are_rejected(self):
        p = Parameter('torch.bfloat16')
        p.grad = object()
        with self.assertRaisesRegex(RuntimeError, 'precede backward'):
            normalize_trainable_precision(SimpleNamespace(named_parameters=lambda: iter([('lora_A', p)])))
        p.grad = None
        with self.assertRaisesRegex(RuntimeError, 'unexpected trainable'):
            normalize_trainable_precision(SimpleNamespace(named_parameters=lambda: iter([('base.weight', p)])))

    def test_retry_accepts_only_empty_private_destination(self):
        for private, files, accepted in [(True, ['.gitattributes'], True),
                                         (False, [], False),
                                         (True, ['adapter/model.safetensors'], False)]:
            api = SimpleNamespace(create_repo=lambda **kw: None,
                repo_info=lambda **kw: SimpleNamespace(private=private),
                list_repo_files=lambda **kw: files)
            if accepted:
                prepare_artifact_repo(api, 'owner/smoke')
            else:
                with self.assertRaises(ValueError):
                    prepare_artifact_repo(api, 'owner/smoke')


if __name__ == '__main__':
    unittest.main()
