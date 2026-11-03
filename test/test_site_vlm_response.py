import json,unittest
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
from pai_ros.cargo_vlm import response_text
from pai_ros.site_perception import vlm_observe,fixture_image,parse_sites,parse_directive,JUDGMENT_PROMPT,JUDGMENT_VERSION

class SiteVLMResponseTests(unittest.TestCase):
    def test_priority_order_cannot_override_fixed_policy(self):
        with self.assertRaises(ValueError):parse_directive(dict(priority_order=[1,2,3],reason='override'))
    def test_case_only_id_variants_are_unambiguous(self):
        result=parse_sites(json.dumps({'PICKUP_a':dict(available=False,reason='blocked'),'meeting_b':dict(available=True,reason='open')}),['pickup_A','meeting_B'])
        self.assertFalse(result['pickup_A']['available'])
        self.assertTrue(result['meeting_B']['available'])
    def test_unknown_missing_or_colliding_ids_remain_invalid(self):
        for keys in (['camera_1'],[],['pickup_A','pickup_a'],['site_pickup_A']):
            with self.assertRaises(ValueError):parse_sites(json.dumps({k:dict(available=True) for k in keys}),['pickup_A'])
    def response(self,text):
        return SimpleNamespace(model='test/vision',choices=[SimpleNamespace(message=SimpleNamespace(content=text),finish_reason='stop')])
    def test_color_normalization_remains_backward_compatible(self):
        self.assertEqual(response_text(self.response(' RED '))[0],'red')
        self.assertEqual(response_text(self.response(' pickup_A '),preserve_case=True)[0],'pickup_A')
    def test_vlm_camera_ids_survive_response_parsing(self):
        images={key:fixture_image(False) for key in ('pickup_A','pickup_B','meeting_A','meeting_B')}
        payload=dict(sites={key:dict(available=True,reason='Open road') for key in images},directive=dict(priority_order=[3,2,1],reason='Prioritize P3'))
        client=MagicMock();client.chat.completions.create.return_value=self.response(json.dumps(payload))
        with patch('openai.OpenAI') as constructor:
            constructor.return_value.__enter__.return_value=client
            result=vlm_observe(images,['test/vision'],'https://openrouter.ai/api/v1','test-key',context={'orders':[]})
        self.assertEqual(set(result['sites']),set(images))
        self.assertEqual(result['directive']['reason'],'Prioritize P3')
        self.assertEqual(result['inference']['requested_model'],'test/vision')
        self.assertGreaterEqual(result['inference']['latency_s'],0)
        messages=client.chat.completions.create.call_args.kwargs['messages']
        self.assertEqual(messages[0],dict(role='system',content=JUDGMENT_PROMPT))
        self.assertEqual(result['inference']['prompt_version'],JUDGMENT_VERSION)
        content=messages[1]['content']
        self.assertEqual(sum(row['type']=='image_url' for row in content),4)
    def test_invalid_response_does_not_fall_back_to_fixture(self):
        client=MagicMock();client.chat.completions.create.return_value=self.response('{}')
        with patch('openai.OpenAI') as constructor:
            constructor.return_value.__enter__.return_value=client
            with self.assertRaises(RuntimeError):vlm_observe({'pickup_A':fixture_image(False)},['test/vision'],'https://openrouter.ai/api/v1','test-key',context={})

if __name__=='__main__':unittest.main()
