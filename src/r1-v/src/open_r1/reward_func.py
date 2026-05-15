import re
import os
import json
import numpy as np
from datetime import datetime
from rouge_score import rouge_scorer
import ast

# ans_acc_reward
# ans_tiou_reward
# ans_viou_reward
# thk_temporal_point_reward
# thk_temporal_segment_reward
# thk_spatial_reward
# format_reward

STRUCTURED_JUDGE_KEYS = (
    "temporal_relation",
    "answer_window_relation",
    "reasoning_answer_consistency",
)


def _get_kwarg_value(kwargs, key, idx):
    values = kwargs.get(key)
    if values is None or idx >= len(values):
        return None
    return values[idx]


def _get_feedback_value(kwargs, idx):
    for key in ("feedback", "process_feedback"):
        value = _get_kwarg_value(kwargs, key, idx)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _get_structured_metadata(kwargs, idx):
    metadata = {}
    for key in STRUCTURED_JUDGE_KEYS:
        value = _get_kwarg_value(kwargs, key, idx)
        if isinstance(value, str) and value.strip():
            metadata[key] = value.strip()
    judge_metadata = _get_kwarg_value(kwargs, "judge_metadata", idx)
    if isinstance(judge_metadata, dict):
        for key in STRUCTURED_JUDGE_KEYS:
            value = judge_metadata.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value.strip()
    return metadata


def _wrap_reward_outputs(rewards, kwargs, *, metadata_factory=None):
    if not kwargs.get("return_metadata", False):
        return rewards

    wrapped = []
    for idx, reward in enumerate(rewards):
        item = {"score": float(reward)}
        if metadata_factory is not None:
            item.update(metadata_factory(float(reward), idx) or {})
        feedback = _get_feedback_value(kwargs, idx)
        if feedback:
            item["feedback"] = feedback
        feedback_seed = _get_kwarg_value(kwargs, "feedback_seed", idx)
        if isinstance(feedback_seed, str) and feedback_seed.strip():
            item["feedback_seed"] = feedback_seed.strip()
        item.update(_get_structured_metadata(kwargs, idx))
        wrapped.append(item)
    return wrapped


def ans_acc_reward(completions, answer, **kwargs):

    solution = [f'<answer>{ans}</answer>' for ans in answer]
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = "free-form"  # temporal-spatial free-form QA/ General video QA Free-form
    
    if kwargs['task'][0] == "temporal QA (MCQ)":
        question_type = "TG_MCQ"

    if kwargs['task'][0] == "General video QA MCQ":
        question_type = "MCQ"
    
    if kwargs['task'][0] == "visual QA" or kwargs['task'][0] == "temporal QA":
        question_type = "none"

    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    idx = 0
    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "TG_MCQ":
                gt_ans = answer[idx].split("\n")[0]
                try:
                    choice = output_ans.split("Correct Option:")[1]
                    gt_ans = gt_ans.strip()
                    gt_list = [gt_ans,  gt_ans+'.',  '(' + gt_ans + ')', '[' + gt_ans + ']']
                    reward = 1.0 if choice.strip() in gt_list else 0.0
                    # msg = f"success when calculating acc: {output_ans}|| reward: + {reward}"
                    # print(msg)
                except:
                    # msg = "error when calculating acc:" + output_ans
                    # print(msg)
                    reward = 0.0
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "MCQ":
                choice = output_ans
                gt_ans = gt_ans.strip()
                gt_list = [gt_ans,  gt_ans+'.',  '(' + gt_ans + ')', '[' + gt_ans + ']']
                reward = 1.0 if choice.strip() in gt_list else 0.0
            else:
                reward = 0.0
            idx += 1
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
            
    return _wrap_reward_outputs(
        rewards,
        kwargs,
        metadata_factory=lambda reward, idx: {"acc_score": reward},
    )

def ans_tiou_reward(completions, answer, **kwargs):

    solution = [f'<answer>{ans}</answer>' for ans in answer]
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    question_type = "none"

    if kwargs['task'][0] == "temporal QA":
        question_type = "TG"
    
    if kwargs['task'][0] == "temporal QA (MCQ)":
        question_type = "TG_MCQ"


    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    idx = 0

    for content, sol in zip(contents, solution):

        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "TG":
                gt_ans = answer[idx]
                gt_ans = ast.literal_eval(gt_ans)
                pattern = r"<t>(\d+\.?\d*)</t>s to <t>(\d+\.?\d*)</t>s"
                reward = 0.0
                match = re.search(pattern, output_ans)
                if match:
                    start_time_str = match.group(1) 
                    end_time_str = match.group(2)
                    start_time = float(start_time_str)
                    end_time = float(end_time_str)

                    if end_time < start_time:
                        times = []
                    else:
                        times = [start_time, end_time]
                else:
                    times = []

                if len(times) == 2:
                    start1, end1 = times
                    start2, end2 = gt_ans
                    intersection_start = max(start1, start2)
                    intersection_end = min(end1, end2)
                    intersection_length = max(0, intersection_end - intersection_start)
                    union_length = max(end1, end2) - min(start1, start2)
                    iou = intersection_length / union_length if union_length != 0 else 0
                    reward = iou
            elif question_type=="TG_MCQ":
                gt_ans = answer[idx]
                gt_ans = gt_ans.split("\n")[1]
                gt_ans = ast.literal_eval(gt_ans)
                pattern = r"<t>(\d+\.?\d*)</t>s to <t>(\d+\.?\d*)</t>s"
                reward = 0.0
                match = re.search(pattern, output_ans)
                if match:
                    start_time_str = match.group(1) 
                    end_time_str = match.group(2)     
                    start_time = float(start_time_str)
                    end_time = float(end_time_str)
                    if end_time < start_time:
                        times = []
                    else:
                        times = [start_time, end_time]
                else:
                    times = []

                if len(times) == 2:
                    start1, end1 = times
                    start2, end2 = gt_ans
                    intersection_start = max(start1, start2)
                    intersection_end = min(end1, end2)
                    intersection_length = max(0, intersection_end - intersection_start)
                    union_length = max(end1, end2) - min(start1, start2)
                    iou = intersection_length / union_length if union_length != 0 else 0
                    reward = iou
            else:
                reward = 0.0
            idx += 1
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
            
    return _wrap_reward_outputs(
        rewards,
        kwargs,
        metadata_factory=lambda reward, idx: {"time_reward_score": reward},
    )


def ans_viou_reward(completions, answer, **kwargs):
    solution = [f'<answer>{ans}</answer>' for ans in answer]
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    question_type = "none"
    if kwargs['task'][0] == "visual QA":
        question_type = "VG"
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    idx = 0

    for content, sol in zip(contents, solution):

        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            
            if question_type == "VG":
                reward = 0.0
                pattern = r"<box>(\[.*?\])</box>"
                match_gt = re.search(pattern, sol)
                if match_gt:
                    bbox_gt = match_gt.group(1)
                    bbox_gt = json.loads(bbox_gt)
                else:
                    bbox_gt = None

                match_pred = re.search(pattern, output_ans)
                if match_pred:
                    bbox_pred = match_pred.group(1)
                    bbox_pred = json.loads(bbox_pred)
                    if bbox_gt is not None and bbox_pred is not None:
                        bbox_gt = convert_coord_format_gqa(bbox_gt, kwargs["image_size"][idx], kwargs["image_size_refine"][idx])
                        reward = calculate_iou(bbox_gt, bbox_pred)
            else:
                reward = 0.0
            idx += 1
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
            
    return _wrap_reward_outputs(
        rewards,
        kwargs,
        metadata_factory=lambda reward, idx: {"spatial_reward_score": reward},
    )


def format_reward(completions, **kwargs):

    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>.*?</answer>"

    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]

    for content in completion_contents:
        think_match = re.search(think_pattern, content, re.DOTALL)
        answer_match = re.search(answer_pattern, content, re.DOTALL)

        if not (think_match and answer_match):
            rewards.append(0.0)
            continue 
        
        count_think_start = content.count("<think>")
        count_think_end = content.count("</think>")
        if count_think_start!= count_think_end:
            rewards.append(0.0)
            continue

        count_answer_start = content.count("<answer>")
        count_answer_end = content.count("</answer>")
        if count_answer_start!= count_answer_end:
            rewards.append(0.0)
            continue

        think_content = think_match.group(1)

        count_obj_start = think_content.count('<obj>')
        count_obj_end = think_content.count('</obj>')
        count_time_start = think_content.count('<t>')
        count_time_end = think_content.count('</t>')
        count_box_start = think_content.count('<box>')
        count_box_end = think_content.count('</box>')

        is_obj_paired = (count_obj_start == count_obj_end)
        is_time_paired = (count_time_start == count_time_end)
        is_box_paired = (count_box_start == count_box_end)

        if not (is_obj_paired and is_time_paired and is_box_paired):
            rewards.append(0.0)
            continue
        
        has_st_reasoning = (count_obj_start > 0 and count_time_start > 0 and count_box_start > 0)

        if kwargs['task'][0] == "temporal QA" or kwargs['task'][0] == "temporal QA (MCQ)":
            has_st_reasoning = count_time_start >= 2
        
        if kwargs['task'][0] == "visual QA":
            pattern = r"<obj>(\w+)</obj><box>(\[.*?\])</box>"
            match_pred = re.search(pattern, content)
            if match_pred:
                has_st_reasoning = True

        if has_st_reasoning or "General video QA" in kwargs['task'][0]:
            rewards.append(1.0)
        else:
            rewards.append(0.5)


    return _wrap_reward_outputs(
        rewards,
        kwargs,
        metadata_factory=lambda reward, idx: {"format_reward_score": reward},
    )


def parse_temporal_spatial_reasoning_process(think_content: str):

    pattern = r"<obj>(.*?)</obj>((?:<box>\[.*?\]</box>)+)at<t>(.*?)</t>s"
    parsed_claims = []
    count = 0

    for match in re.finditer(pattern, think_content, re.DOTALL):
        try:
            object_name = match.group(1).strip()
            all_boxes_str = match.group(2)  
            timestamp_str = match.group(3).strip()
            timestamp = float(timestamp_str)
            
            individual_box_strs = re.findall(r'\[.*?\]', all_boxes_str)
            bboxes = [json.loads(b_str) for b_str in individual_box_strs]
            
            parsed_claims.append({
                "id": count,
                "object_name": object_name,
                "timestamp": timestamp,
                "bboxes": bboxes
            })
            count += 1

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            continue
            
    return parsed_claims

def convert_coord_format(bbox, image_size):
    nx_min, ny_min, nx_max, ny_max = bbox
    width, height = image_size
    x_min = nx_min * width
    y_min = ny_min * height
    x_max = nx_max * width
    y_max = ny_max * height

    return [x_min, y_min, x_max, y_max]


def convert_coord_format_gqa(bbox, image_size, image_size_refine):
    bbox[0] = bbox[0] * image_size_refine[0] / image_size[0]
    bbox[1] = bbox[1] * image_size_refine[1] / image_size[1]
    bbox[2] = bbox[2] * image_size_refine[0] / image_size[0]
    bbox[3] = bbox[3] * image_size_refine[1] / image_size[1]
    return bbox

def name_match(pred_name, gt_name, mode="soft"):
    pred = pred_name.lower().strip()
    gt = gt_name.lower().strip()
    
    if mode == "strict":
        return pred == gt
    
    if pred == gt:
        return True

    if pred in gt or gt in pred:
        return True
    
    pred_words = set(pred.split())
    gt_words = set(gt.split())
    if pred_words & gt_words:  
        return True
    
    return False


def calculate_iou(boxA, boxB):

    try:
        if not (isinstance(boxB, list) and len(boxB) == 4):
            return 0.0

        boxA_corners = np.array(boxA, dtype=float)
        boxB_corners = np.array(boxB, dtype=float)

    except (ValueError, TypeError, IndexError):
        return 0.0

    xA = max(boxA_corners[0], boxB_corners[0])
    yA = max(boxA_corners[1], boxB_corners[1])
    xB = min(boxA_corners[2], boxB_corners[2])
    yB = min(boxA_corners[3], boxB_corners[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA_corners[2] - boxA_corners[0]) * (boxA_corners[3] - boxA_corners[1])
    boxB_area = (boxB_corners[2] - boxB_corners[0]) * (boxB_corners[3] - boxB_corners[1])

    union_area = boxA_area + boxB_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def thk_temporal_segment_reward(completions, **kwargs):

    temporal_reward = []
    idx = 0

    for completion in completions:
        think_match = re.search(r"<think>(.*?)</think>", completion[0]["content"], re.DOTALL)

        if not think_match or kwargs['task'][0] == "visual QA" or kwargs['task'][0] == "temporal-spatial free-form QA" or "General video QA" in kwargs['task'][0]:
            temporal_reward.append(0.0)
            idx += 1
            continue

        think_content = think_match.group(1)
        reward = 0.0
        pattern = r'<t>([\d.]+)</t>s'
        gt_ans = kwargs["answer"][idx]
        if kwargs['task'][0] == "temporal QA (MCQ)":
            gt_ans = gt_ans.split("\n")[1]
        gt_ans = ast.literal_eval(gt_ans)

        try: 
            times = [float(match) for match in re.findall(pattern, think_content)]
        except:
            times = []
            reward = 0.0

        if len(times) > 0:
            for pred_time in times:
                if gt_ans[0] <= pred_time <= gt_ans[1]:
                    reward += 1.0
            reward = reward/len(times)

        temporal_reward.append(reward)
        idx += 1

    return _wrap_reward_outputs(
        temporal_reward,
        kwargs,
        metadata_factory=lambda reward, idx: {"time_reward_score": reward},
    )


def thk_temporal_point_reward(completions, **kwargs):

    step_percent = kwargs['step_percent'][0]

    temporal_reward = []
    idx = 0

    for completion in completions:
        think_match = re.search(r"<think>(.*?)</think>", completion[0]["content"], re.DOTALL)

        if not think_match or kwargs['task'][0] in ["visual QA", "temporal QA", "temporal QA (MCQ)"] or "General video QA" in kwargs['task'][0]:
            temporal_reward.append(0.0)
            idx += 1
            continue

        think_content = think_match.group(1)

        reward = 0.0
        pattern = r'<t>([\d.]+)</t>s'
        try: 
            pred_times = [float(match) for match in re.findall(pattern, think_content)]
        except:
            pred_times = []

        if len(pred_times) > 0:
            gt_times = [frame["time"] for frame in kwargs["key_frames"][idx]]
            total_proximity_score = 0.0
            for time in pred_times:
                time_diff = min([abs(time - gt_time) for gt_time in gt_times])
                if step_percent < 3/4:
                    sigma = 4*(1-step_percent) 
                else:
                    sigma = 1
                proximity_score = np.exp(-(time_diff ** 2) / (2 * sigma ** 2))
                total_proximity_score += proximity_score
            
            temporal_reward.append(total_proximity_score / len(pred_times))
        else:
            temporal_reward.append(0.0)
        idx += 1

    return _wrap_reward_outputs(
        temporal_reward,
        kwargs,
        metadata_factory=lambda reward, idx: {"time_reward_score": reward},
    )


# thinking -> spatial reward 
def thk_spatial_reward(completions, **kwargs):
    
    spatial_reward = []
    idx = 0

    for completion in completions:
        think_match = re.search(r"<think>(.*?)</think>", completion[0]["content"], re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", completion[0]["content"], re.DOTALL)

        if not think_match or not answer_match:
            spatial_reward.append(0.0)
            idx += 1
            continue

        '''
        < visual QA example > 
        question: Who is wearing the bag?
        "answer": "<obj>pedestrian</obj><box>[421, 187, 443, 239]</box>",
        '''
        if kwargs['task'][0] == "visual QA":

            pattern = r"<box>(\[.*?\])</box>"
            match_gt = re.search(pattern, kwargs["answer"][idx])
            bbox_gt = None
            if match_gt:  
                try:
                    bbox_gt = match_gt.group(1)       
                    bbox_gt = json.loads(bbox_gt)
                except:
                    bbox_gt = None
            
            # think part
            output_think = think_match.group(1)

            match_pred = re.findall(pattern, output_think)
            bboxes_pred = []
            if match_pred:
                for bbox_item in match_pred:
                    bbox = bbox_item
                    try:
                        bboxes_pred.append(json.loads(bbox))
                    except:
                        pass
            if len(bboxes_pred) > 0 and bbox_gt is not None:
                max_iou = 0.0
                bbox_gt = convert_coord_format_gqa(bbox_gt, kwargs["image_size"][idx], kwargs["image_size_refine"][idx])
                for bbox_pred in bboxes_pred:
                    iou = calculate_iou(bbox_gt, bbox_pred)
                    max_iou = max(max_iou, iou)
                spatial_reward.append(max_iou)
            else:
                spatial_reward.append(0.0)
            
            idx += 1
            continue

        if kwargs['task'][0] == "temporal QA" or kwargs['task'][0] == "temporal QA (MCQ)" or "General video QA" in kwargs['task'][0]:
            spatial_reward.append(0.0)
            idx += 1
            continue
        think_content = think_match.group(1)
        parsed_claims = parse_temporal_spatial_reasoning_process(think_content)

        if not parsed_claims:
            spatial_reward.append(0.0)
            idx += 1
            continue

        gt_items = kwargs["key_items"][idx]
        gt_times = [frame["time"] for frame in kwargs["key_frames"][idx]]

        total_iou_score = 0.0
        matched_claims = 0  

        for claim in parsed_claims:
            pred_time = claim['timestamp']
            closest_time = -1
            min_time_diff = float('inf')

            threshold = 1.0  
            
            correct_tempgate = kwargs.get("correct_tempgate", True)

            for ii in range(len(gt_times)):
                if correct_tempgate:
                    time_condition = abs(gt_times[ii] - pred_time) <= threshold
                else:
                    time_condition = gt_times[ii] - pred_time < threshold
                
                if time_condition:
                    time_diff = abs(gt_times[ii] - pred_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_time = gt_times[ii]
            if closest_time == -1:
                continue
            
            key_frame = None
            for ii in range(len(kwargs["key_frames"][idx])):
                if kwargs["key_frames"][idx][ii]["time"] == closest_time:
                    key_frame = kwargs["key_frames"][idx][ii]
                    break

            if claim['bboxes'] is not None and isinstance(claim['bboxes'], list) and key_frame is not None:
                objects = gt_items[str(key_frame["idx"])]
                spatial_iou_mode = kwargs.get("spatial_iou_mode", "max")
                identity_match_mode = kwargs.get("identity_match_mode", "none")
                object_ious = [] 
                
                for obj in objects.keys():
                    if identity_match_mode != "none":
                        pred_obj_name = claim.get('object_name', '')
                        if not name_match(pred_obj_name, obj, mode=identity_match_mode):
                            continue  
                    
                    claim_boxes = claim['bboxes']
                    gt_boxes = objects[obj]       
                    try:
                        is_claim_originally_multiple = isinstance(claim_boxes[0], list)
                    except:
                        print("Error:", claim_boxes)
                        continue
                        
                    if not is_claim_originally_multiple:
                        claim_boxes = [claim_boxes]
                
                    list_of_max_ious = [] 

                    for gt_box in gt_boxes:
                        gt_box = convert_coord_format(gt_box, kwargs["image_size"][idx])
                        ious_for_current_gt = [calculate_iou(gt_box, c_box) for c_box in claim_boxes]
                        iou_for_gt = max(ious_for_current_gt) if ious_for_current_gt else 0.0    
                        list_of_max_ious.append(iou_for_gt)

                    if list_of_max_ious:
                        iou = sum(list_of_max_ious) / len(list_of_max_ious)
                        object_ious.append(iou)
                

                if object_ious:
                    matched_claims += 1  
                    if spatial_iou_mode == "avg":
                        total_iou_score += sum(object_ious) / len(object_ious)
                    else:  
                        total_iou_score += max(object_ious)

        spatial_norm_mode = kwargs.get("spatial_norm_mode", "all")
        if spatial_norm_mode == "matched":
            denominator = max(1, matched_claims)
        else: 
            denominator = len(parsed_claims)
        
        spatial_reward.append(total_iou_score / denominator)
        idx += 1

    return _wrap_reward_outputs(
        spatial_reward,
        kwargs,
        metadata_factory=lambda reward, idx: {"spatial_reward_score": reward},
    )
