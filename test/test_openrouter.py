import os
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import numpy as np
from pai_ros.cargo_vlm import classify_vlm, vlm_decide, api_key_for, DEFAULT_MODEL, DEFAULT_BASE_URL, PROMPT

def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

class OpenRouterTest(unittest.TestCase):
    def test_existing_prompt_model_and_image_sent(self):
        with patch('openai.OpenAI') as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.completions.create.return_value = response('blue')
            result = vlm_decide(np.zeros((2,2,3), dtype=np.uint8), [DEFAULT_MODEL], DEFAULT_BASE_URL, 'test-key')
            self.assertEqual(result[0], 'blue')
            self.assertEqual(factory.call_args.kwargs['base_url'], DEFAULT_BASE_URL)
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs['model'], DEFAULT_MODEL)
            self.assertEqual(kwargs['messages'][0]['content'][0]['text'], PROMPT)
            self.assertTrue(kwargs['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))

    def test_ambiguous_answer_falls_back(self):
        with patch('openai.OpenAI') as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.completions.create.side_effect = [response('red, green, or blue'), response('green')]
            self.assertEqual(classify_vlm(np.zeros((2,2,3),dtype=np.uint8), ['first','second'], DEFAULT_BASE_URL, 'test'), 'green')
            self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_missing_key_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
                api_key_for(DEFAULT_BASE_URL)
            self.assertEqual(api_key_for('http://localhost:11434/v1'), 'ollama')

if __name__ == '__main__': unittest.main()

class DiagnosticTest(unittest.TestCase):
    def test_api_failure_retains_status_without_key(self):
        class Rejected(Exception):
            status_code = 401
        with patch('openai.OpenAI') as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.completions.create.side_effect = Rejected('bad key sk-secret-example')
            with self.assertRaises(RuntimeError) as ctx:
                classify_vlm(np.zeros((2,2,3),dtype=np.uint8), ['test-model'], DEFAULT_BASE_URL, 'sk-secret-example')
            self.assertIn('401', str(ctx.exception))
            self.assertNotIn('sk-secret-example', str(ctx.exception))

    def test_empty_response_distinguished(self):
        with patch('openai.OpenAI') as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.completions.create.return_value = response('')
            with self.assertRaisesRegex(RuntimeError, 'empty content'):
                classify_vlm(np.zeros((2,2,3),dtype=np.uint8), ['test-model'], DEFAULT_BASE_URL, 'test-key')

class EnvelopeTest(unittest.TestCase):
    def test_null_choices_preserves_provider_error(self):
        from pai_ros.cargo_vlm import response_text
        with self.assertRaisesRegex(RuntimeError, '429.*rate limited'):
            response_text(SimpleNamespace(choices=None, error={'code':429, 'message':'rate limited'}))

    def test_missing_choices_is_not_typeerror(self):
        from pai_ros.cargo_vlm import response_text
        for choices in (None, []):
            with self.assertRaisesRegex(ValueError, 'no choices'):
                response_text(SimpleNamespace(choices=choices))

    def test_error_in_sdk_extras(self):
        from pai_ros.cargo_vlm import response_text
        with self.assertRaisesRegex(RuntimeError, '503'):
            response_text(SimpleNamespace(choices=None, model_extra={'error':{'code':503, 'message':'provider unavailable'}}))

class RetryTest(unittest.TestCase):
    def test_busy_provider_then_success(self):
        from pai_ros.cargo_vlm import TemporaryVLMError
        with patch('pai_ros.cargo_vlm._classify_vlm_once', side_effect=[TemporaryVLMError('busy'), 'red']) as call, patch('pai_ros.cargo_vlm.time.sleep') as sleep:
            self.assertEqual(classify_vlm(None, ['model'], DEFAULT_BASE_URL, 'key'), 'red')
            self.assertEqual(call.call_count, 2)
            sleep.assert_called_once_with(2)

    def test_retry_is_bounded(self):
        from pai_ros.cargo_vlm import TemporaryVLMError
        with patch('pai_ros.cargo_vlm._classify_vlm_once', side_effect=TemporaryVLMError('busy')) as call, patch('pai_ros.cargo_vlm.time.sleep') as sleep:
            with self.assertRaises(TemporaryVLMError): classify_vlm(None, ['model'], DEFAULT_BASE_URL, 'key')
            self.assertEqual(call.call_count, 3)
            self.assertEqual([x.args[0] for x in sleep.call_args_list], [2, 4])

    def test_auth_failure_does_not_retry(self):
        with patch('pai_ros.cargo_vlm._classify_vlm_once', side_effect=RuntimeError('401')) as call, patch('pai_ros.cargo_vlm.time.sleep') as sleep:
            with self.assertRaises(RuntimeError): classify_vlm(None, ['model'], DEFAULT_BASE_URL, 'key')
            self.assertEqual(call.call_count, 1)
            sleep.assert_not_called()
