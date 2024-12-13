verbose = False

import copy
import random
import asyncio
import math
import os
import numpy as np


from conversationgenome.utils.Utils import Utils
from conversationgenome.ConfigLib import c

from conversationgenome.miner.MinerLib import MinerLib
from conversationgenome.conversation.ConvoLib import ConvoLib
from conversationgenome.llm.LlmLib import LlmLib
from conversationgenome.mock.MockBt import MockBt

bt = None
try:
    import bittensor as bt
except:
    if verbose:
        print("bittensor not installed")
    bt = MockBt()

if c.get('env', 'FORCE_LOG') == 'debug':
    bt.logging.enable_debug(True)
elif c.get('env', 'FORCE_LOG') == 'info':
    bt.logging.enable_default(True)
try:
    import wandb
except Exception as e:
    print("Wand error")

proto = {
    "interests_of_q": [],
    "hobbies_of_q": [],
    "personality_traits_of_q": [],
    "interests_of_a": [],
    "hobbies_of_a": [],
    "personality_traits_of_a": [],
}


class ValidatorLib:
    mode = "test"
    hotkey = "v1234"
    verbose = False
    llml = None

    def __init__(self):
        super(ValidatorLib, self).__init__()


    async def reserve_conversation(self, minConvWindows = 1, batch_num=None):
        import time
        out = None
        full_conversation = await self.getConvo()
        if self.verbose:
            bt.logging.info("full_conversation", full_conversation)

        if full_conversation:
            conversation_guid = str(Utils.get(full_conversation, "guid"))
            num_lines = len(Utils.get(full_conversation, 'lines', []))
            llm_type = "openai"
            model = "gpt-4o"
            llm_type_override = c.get("env", "LLM_TYPE_OVERRIDE")
            if llm_type_override:
                llm_type = llm_type_override
                model = c.get("env", "OPENAI_MODEL")

            bt.logging.info(f"Reserved conversation ID: {conversation_guid} with {num_lines} lines. Sending to {llm_type}:{model} LLM...")

            full_conversation_metadata = await self.generate_full_convo_metadata(full_conversation)
            if not full_conversation_metadata:
                bt.logging.error(f"ERROR:927402. No metadata for conversation returned to validator. Aborting.")
                validatorHotkey = "HK-FAIL"
                await self.put_convo("NO-TAGS", conversation_guid, {"tags":[], "vectors":[]}, type="validator", batch_num=batch_num)

                return None
            full_conversation_tags = Utils.get(full_conversation_metadata, "tags", [])
            full_conversation_vectors = Utils.get(full_conversation_metadata, "vectors", [])
            bt.logging.info(f"Found {len(full_conversation_tags)} tags and {len(full_conversation_vectors)} in FullConvo")

            log_path = c.get('env', 'SCORING_DEBUG_LOG')
            if not Utils.empty(log_path):
                Utils.append_log(log_path, f"Validator found full convo tags {full_conversation_tags} in FullConvo")

            minValidTags = self.validateMinimumTags(full_conversation_tags)
            if minValidTags:
                convoWindows = self.getConvoWindows(full_conversation)
                if len(convoWindows) > minConvWindows:
                    out = (full_conversation, full_conversation_metadata, convoWindows)
                else:
                    bt.logging.info(f"Not enough convo windows -- only {len(convoWindows)}. Passing.")
                    out = None
            else:
                bt.logging.info("Not enough valid tags for conversation. Passing.")
                out = None
            return out
        else:
            bt.logging.error(f"ERROR:9879432: No conversation returned from API. Aborting.")
        return None

    async def getConvo(self):
        hotkey = self.hotkey
        cl = ConvoLib()
        convo = await cl.get_conversation(hotkey)
        return convo

    async def put_convo(self, hotkey, c_guid, data, type="validator", batch_num=None, window=None):
        cl = ConvoLib()
        convo = await cl.put_conversation(hotkey, c_guid, data, type=type, batch_num=batch_num, window=window)
        return convo


    def getConvoWindows(self, fullConvo):
        minLines = c.get("convo_window", "min_lines", 5)
        maxLines = c.get("convo_window", "max_lines", 10)
        overlapLines = c.get("convo_window", "overlap_lines", 2)

        windows = Utils.split_overlap_array(fullConvo['lines'], size=maxLines, overlap=overlapLines)
        if len(windows) < 2:
            windows = Utils.split_overlap_array(fullConvo['lines'], size=minLines, overlap=overlapLines)

        return windows

    async def filter_valid_tags(self, tags):
        return tags


    async def generate_full_convo_metadata(self, convo):
        if self.verbose:
            bt.logging.info(f"Execute generate_full_convo_metadata for participants {convo['participants']}")
        else:
            bt.logging.info(f"Execute generate_full_convo_metadata")

        llml = LlmLib()
        self.llml = llml
        result = await llml.conversation_to_metadata(convo, generateEmbeddings=True)
        if not result:
            bt.logging.error(f"ERROR:2873226353. No conversation metadata returned. Aborting.")
            return None
        if not Utils.get(result, 'success'):
            bt.logging.error(f"ERROR:2873226354. Conversation metadata failed: {result}. Aborting.")
            return None

        tags = result['tags']
        vectors = Utils.get(result, 'vectors', {})
        data = {
            "participantProfiles": convo['participants'],
            "tags": tags,
            "vectors": vectors,
        }
        return data

    async def get_vector_embeddings_set(self, tags):
        response = await self.llml.get_vector_embeddings_set(tags)
        return response


    async def send_to_miners(self, conversation_guid, window_idx, conversation_window, miner_uids):
        bt.logging.info(f"Send to conversation {conversation_guid} / {window_idx} to miners: {miner_uids}")
        results = []
        ml = MinerLib()
        tasks = [asyncio.create_task(ml.do_mining(conversation_guid, window_idx, conversation_window, minerUid)) for minerUid in miner_uids]
        await asyncio.wait(tasks)
        for task in tasks:
            results.append(task.result())
        return results

    def validateMinimumTags(self, tags):
        return True

    def selectStage1Miners(self, uids, num=3):
        selectedMiners = random.sample(uids, num)
        return selectedMiners

    async def outputEmissions(self, convoId, windowId, emissionRewards):
        bt.logging.info("EMISSIONS for %d window %d" % (convoId, windowId), emissionRewards)

    async def send_windows_to_test_miners(self, windows, full_conversation=None, full_conversation_metadata=None):
        conversation_guid = Utils.get(full_conversation, "uid")
        participantProfiles = Utils.get(full_conversation_metadata, "participantProfiles", [])
        full_conversationTags = Utils.get(full_conversation_metadata, "tags", [])
        full_conversationTagVectors = Utils.get(full_conversation_metadata, "tag_vectors", {})

        if self.verbose:
            bt.logging.info("full_conversationTagVectors", full_conversationTagVectors)
        vectorNeightborhood = []
        for key, full_conversationTagVector in full_conversationTagVectors.items():
            vectorNeightborhood.append(full_conversationTagVector['vectors'])

        semantic_neighborhood = np.mean(vectorNeightborhood, axis=0)

        if self.verbose:
            bt.logging.info("Full convo tags", full_conversationTags)

        success = True
        for idx, window in enumerate(windows):
            minersPerWindow = c.get("validator", "miners_per_window", 3)
            uids = [1,2,3,4,5,6,7,8,9]
            miners = self.selectStage1Miners(uids, minersPerWindow)
            miner_results = await self.send_to_miners(conversation_guid, idx, window, miners)

            for minerResult in minerResults:
                uid = Utils.get(minerResult, 'uid')
                tags = Utils.get(minerResult, 'tags')
                bt.logging.info(f"Generate vectors from {len(tags)} miner tags")

                vectors = Utils.get(minerResult, 'vectors')
                compareResults = Utils.compare_arrays(full_conversationTags, tags)
                compareResults['total_1'] = len(full_conversationTags)
                compareResults['total_2'] = len(tags)
                scoreToFullConvo = await self.calculate_base_score(compareResults)
                minerResult['score'] = scoreToFullConvo
                similarity_scores = []
                uniqueTags = compareResults['unique_2']
                if len(uniqueTags) > 0:
                    for unique_tag in uniqueTags:
                        if unique_tag in vectors:
                            tagVectors = vectors[unique_tag]['vectors']
                            similarity_score = 0
                            if not Utils.is_empty_vector(tagVectors):
                                similarity_score = np.dot(semantic_neighborhood, tagVectors) / (np.linalg.norm(semantic_neighborhood) * np.linalg.norm(tagVectors))
                            similarity_scores.append(similarity_score)
                    bt.logging.info("MEDIAN similarity_score of %d unique tags for miner %s" % (len(uniqueTags), str(uid)), np.median(similarity_scores), similarity_scores)
                else:
                    bt.logging.info( "No unique tags for miner %s" % (str(uid)) )

            await self.calculate_emission_rewards(minerResults, 'score')

            rewards = {}
            for minerResult in minerResults:
                rewards[minerResult['uid']] = minerResult['reward']
            await self.outputEmissions(1, idx, rewards)

        if success == True:
            cl = ConvoLib()
            await cl.markConversionComplete(self.hotkey, cguid)

    async def neighborhood_test(self):
        bt.logging.info("Quick test for semantic neighborhood with vectors")
        llml = LlmLib()
        await llml.test_neighborhood()

    async def llm_test(self):
        bt.logging.info("Quick test for LLM")
        llml = LlmLib()
        await llml.test_tagging()


    def update_scores(self, rewards, uids, ema_scores, scores, moving_average_alpha, neurons, nonlinear_power):
        if len(uids) == 0:
            return scores, ema_scores

        rewards = np.array(rewards)
        rewards = np.nan_to_num(rewards, 0.0)
        rewards = np.clip(rewards, 0.0, 1.0)

        scattered_rewards = np.zeros_like(ema_scores)
        for i, uid in enumerate(uids):
            scattered_rewards[uid] = rewards[i]

        ema_scores = (1 - moving_average_alpha) * ema_scores + moving_average_alpha * scattered_rewards

        if np.sum(ema_scores) > 0:
            normalized_scores = ema_scores / np.sum(ema_scores)
        else:
            normalized_scores = np.ones_like(ema_scores) / neurons

        transformed_scores = np.power(normalized_scores, nonlinear_power)

        if np.sum(transformed_scores) > 0:
            scores = transformed_scores / np.sum(transformed_scores)
        else:
            scores = np.ones_like(transformed_scores) / neurons

        return scores, ema_scores

    async def prompt_call_csv(self, convoXmlStr=None, participants=None, override_prompt=None):
        llml = LlmLib()
        return await llml.prompt_call_csv(convoXmlStr, participants, override_prompt)

    async def validate_tag_set(self, originalTagList):
        cleanTagList = Utils.get_clean_tag_set(originalTagList)
        if self.verbose:
            print("Original tag set len: %d clean tag set len: %d" % (len(originalTagList), len(cleanTagList)))
        cleanTagsStr = ",".join(cleanTagList)

        prompt1 = "Separate these keywords into 2 groups: good English keywords and malformed keywords. Malformed keywords should include combined/compound words that are not in the English Dictionary, abbreviations, and typos. Return two comma-delimited lists."
        prompt1 += "\n\n<keywords>\n%s\n</keywords>\n\n" % (cleanTagsStr)

        response = await self.prompt_call_csv(override_prompt=prompt1)
        if len(response['content']) == 0:
            print("EMPTY RESPONSE -- no valid tags", response['content'])
            return None
        contentStr = response['content'].lower()
        goodPos = contentStr.find("good")
        malformedPos = contentStr.find("malformed")
        goodKeywordsStr = contentStr[0:malformedPos].replace("good english keywords:", "").replace("***","").replace("\n","").strip()
        validTags = goodKeywordsStr.split(",")
        validTags = Utils.get_clean_tag_set(validTags)

        processed_tag_list = [element for element in validTags if element in cleanTagsStr]

        return processed_tag_list


