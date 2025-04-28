import base64
import time
from openai import OpenAI
import openai
import os
import json
import csv
from tqdm import tqdm
import fitz
import numpy as np
import concurrent.futures
from datetime import datetime
import multiprocessing as mp
import argparse
import random

#http://35.220.164.252:3888/v1
#https://boyuerichdata.chatgptten.com/v1/"

class ModelEvaluator:
    def __init__(self, model_name, api_key, base_url="https://boyuerichdata.chatgptten.com/v1/", pdf_path="evaluation_doc.pdf"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        # self.doc_content = self._read_pdf(pdf_path)
        
        with open('./data/example.json','r') as f:
            example_data = json.load(f)

        self.example_prompt = f"""

# Instructions for Your Evaluation

- Use the examples above to understand how human evaluators assess 3D models, focusing on what aspects they emphasize.
- Reflect deeply on where evaluation models' reasoning and the human evaluators' assessments diverge and analyze what the humans are keying in on.
- Calibrate your scoring based on those key differences to align closely with the "Human Annotation Ground Truth Scores." The goal is to mirror the human scoring standards.
- Review evaluation models' identified biases and actively prevent them from influencing your analysis.
- Be transparent about your process. Detail any steps you take to avoid human biases and to better align with the human evaluation standards, and include your confidence level for the assessment.

# Example for 3D Model Evaluation

Below is 3 examples, structed in json format, of human annotation ground truth scores and model-generated evaluation scores for three 3D models, along with detailed analysis and evidences. 

{example_data}

Reflection on evaluation models's Evaluation vs. Human Scoring Standards:
Consider the evaluation models's explanation for its scores and explicitly contrast it with the human scoring standards in the "Human Annotation Ground Truth Scores."
Ask yourself:
- What are the key differences in evaluation models's and the human evaluators' scoring criteria?
- Are there aspects valued by human evaluators that evaluation models are underrating, or vice versa?
- How can you adjust your scoring to better reflect the human evaluators' priorities and sensitivities?
- What elements do the humans particularly value, and how can you identify and reward these features?
- How can you improve your evaluation to better capture the nuances and subtleties that humans find important in 3D models?

---

Remember: Your objective is to align with human scoring standards and emulate human expert judgment as closely as possible, explicitly avoiding biases identified by an evaluation model and prioritizing features humans value.

"""
    
    def _fix_json_quotes(self, content):
        result = []
        in_string = False
        escape_next = False
        
        for char in content:
            if escape_next:
                result.append(char)
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                result.append(char)
                continue
                
            if char == '"' and not in_string:
                in_string = True
                result.append(char)
            elif char == '"' and in_string:
                in_string = False
                result.append(char)
            else:
                result.append(char)
        
        if in_string:
            result.append('"')
        
        return ''.join(result)

    def _fix_common_json_issues(self, content):
        content = ''.join(char for char in content if ord(char) >= 32)
        
        content = content.replace('",}', '"}')
        content = content.replace('",]', '"]')
        content = content.replace(',}', '}')
        content = content.replace(',]', ']')
        content = content.replace('}.}', '}}')
        content = content.replace('}.]}', '}]}')
        
        import re
        content = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
        
        return content

    def _read_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            content = ""
            for page in doc:
                content += page.get_text()
            return content
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _encode_video(self, video_path):
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')

    def evaluate_video(self, reference_image_path, video_path, prompt_text=''):
        max_retries = 3
        retry_count = 0
        start_time = time.time()
        
        while retry_count < max_retries:
            try:
                base64_video = self._encode_video(video_path)
                top_p = 0.8 if self.model_name != 'o3' and self.model_name != 'o4-mini' else None
                temperature = 0.3 if self.model_name != 'o4-mini' else None                

                additional_prompt = f'The video shows multiple views of a 3D model generated from the text prompt: "{prompt_text}".' if prompt_text else 'The video shows multiple views of a 3D model generated from the reference image.'
                evaluation_prompt = self.get_evaluation_prompt(additional_prompt=additional_prompt)

                if reference_image_path and not prompt_text:
                    base64_image = self._encode_image(reference_image_path)

                info = f'text prompt: "{prompt_text}"' if prompt_text else 'reference image'

                system_prompt = f"""
# Role and Objective
You are a strict expert 3D model evaluation system with deep expertise in material analysis, and visual quality assessment. This evaluation concerns a video of a 3D model generated from the provided {info}. Your analysis requires the highest level of scrutiny. Deduct points for even minor imperfections without hesitation. Only models demonstrating near-flawless execution across all criteria deserve top scores.

# Video Content Description
The video presents a 3D model rotating smoothly from left to right, showcasing seven distinct view types:
1. Geometry View: Shows structural and geometric information
2. RGB View: Shows colors, materials and textures
3. Lighting View: Shows textures under top lighting
4. Lighting View: Shows textures under right lighting with camera angle down
5. Lighting View: Shows textures under right lighting with camera angle up
6. Lighting View: Shows textures under indoor environment lighting
7. Lighting View: Shows textures under outdoor environment lighting

# Evaluation Instructions
- Analyze the video meticulously, examining each view type in detail.
- Assess the model's material, texture, and visual appearance critically.
- Identify and document any flaws, inconsistencies, or artifacts.
- Provide clear, evidence-backed reasoning for every deduction or score.
- Maintain a professional and objective tone throughout your evaluation.
- Demand excellence in every aspect of the evaluation and score accordingly.

{self.example_prompt}

Begin your detailed evaluation now.                    
"""

                content = [
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{base64_video}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": evaluation_prompt
                            }
                        ] if not prompt_text else [
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{base64_video}"
                                }
                            },
                            {
                                "type": "text",
                                "text": evaluation_prompt
                            }
                        ]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    temperature=temperature,
                    top_p=top_p
                )

                print(f"\nModel {self.model_name} response:")
                print(response.choices[0].message.content)
                result = self._parse_response(response)
                end_time = time.time()
                evaluation_time = end_time - start_time
                print(f"Model {self.model_name} video evaluation completed in {evaluation_time:.2f} seconds.")
                return result[0], result[1], evaluation_time

            except Exception as e:
                print(f"Error in {self.model_name} evaluation (attempt {retry_count}/{max_retries}): {e}")
                print(f"Full error details: {str(e)}")
                random_model = ['gemini-2.0-flash','gemini-2.5-pro-exp-03-25','gemini-2.0-pro'][retry_count % 3]
                retry_count += 1
                self.model_name = random_model
                print (f"Switching to {self.model_name} for retry {retry_count}/{max_retries}...")
                
                if retry_count < max_retries:
                    print(f"Retrying... ({retry_count}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    print(f"Max retries ({max_retries}) reached. Giving up.")
                    return None, None, 0

    def evaluate_image(self, image_path, prompt_text=''):
        max_retries = 3
        retry_count = 0
        start_time = time.time()
        
        while retry_count < max_retries:
            try:
                base64_image = self._encode_image(image_path)
                top_p = 0.8 if self.model_name != 'o3' and self.model_name != 'o4-mini' else None
                temperature = 0.3 if self.model_name != 'o4-mini' else None  
                ref_image_prompt = ' Original reference image / Lighting condition image (leftmost) and' if not prompt_text else ' Lighting condition image (leftmost) and'
                additional_prompt = f'The image shows a 3D model generated from the text prompt: "{prompt_text}".' if prompt_text else ''
                system_prompt = f"""# Role and Objective
    You are an expert 3D model evaluation system with extensive experience in material analysis, and visual quality assessment. Your goal is to provide precise, consistent, and well-reasoned evaluations of 3D models.

    # Core Instructions
    - Maintain extremely high standards for evaluation
    - Identify and penalize even minor imperfections
    - Follow a systematic analysis approach for each criterion
    - Provide clear evidence for all assessments
    - Think step-by-step through each evaluation dimension

    # Analysis Framework
    1. Examine each view type systematically:
    - Geometry View: Structural and geometric information
    - RGB View: Colors, materials and textures
    - Light View: Reflection effects under lights

    2. The provided image shows seven rows of views:
    2.1 First row (Geometry View): Shows structural and geometric information
    2.2 Second row (RGB View): Shows colors, materials and textures
    2.3 Third row (Lighting View): Shows textures under top lighting
    2.4 Fourth row (Lighting View): Shows textures under right lighting with camera angle down
    2.5 Fifth row (Lighting View): Shows textures under right lighting with camera angle up
    2.6 Sixth row (Lighting View): Shows textures under indoor environment lighting
    2.7 Seventh row (Lighting View): Shows textures under outdoor environment lighting
    Each row contains:{ref_image_prompt} Four different views of the 3D model (front, right, back, left)
    
    3. For each view:
    - Analyze front, right, back, and left perspectives
    - Document specific observations
    - Note both strengths and weaknesses
    - Consider relationships between different views
       - Detect artifacts, distortions, or deviations from expected appearance.

    # Human vs Model Evaluation Alignment
    - Understand that model-generated evaluations (`model_evaluations`) may have systematic biases compared to human expert annotations (`human_annotation_ground_truth`).
    - Human annotations reflect nuanced perceptual judgments, contextual understanding, and tolerance to minor imperfections.
    - Model evaluations may over-penalize or under-penalize certain artifacts due to reliance on automated metrics.
    - Adjust your scoring logic to bridge this gap by:
    - Emphasizing perceptual quality and semantic correctness over purely numerical discrepancies.
    - Incorporating tolerance for minor boundary or texture variations that humans typically overlook.
    - Highlighting critical flaws that humans would consider significant, even if metrics show small deviations.
    - Your goal is to produce evaluation scores that better correlate with human ground truth, improving reliability for downstream model ranking and selection.
    
    {self.example_prompt}

    # Response Rules
    - Always provide detailed reasoning before scoring
    - Support each score with specific evidence
    - Be consistent in applying scoring criteria
    - Maintain professional and objective tone
    - Challenge your own assumptions
    - Verify assessments before finalizing

    Remember: Your evaluation must be thorough, evidence-based, and maintain the highest standards of accuracy."""
                evaluation_prompt = self.get_evaluation_prompt(additional_prompt=additional_prompt)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": evaluation_prompt
                                }
                            ]
                        }
                    ],
                    temperature=temperature,
                    top_p=top_p
                )
            
                print(f"\nModel {self.model_name} response:")
                print(response.choices[0].message.content)
            
                initial_scores, initial_explanations = self._parse_response(response)
                end_time = time.time()
                evaluation_time = end_time - start_time
                print(f"Model {self.model_name} evaluation completed in {evaluation_time:.2f} seconds.")
                return initial_scores, initial_explanations, evaluation_time            
            except Exception as e:
                retry_count += 1
                print(f"Error in {self.model_name} evaluation (attempt {retry_count}/{max_retries}): {e}")
                print(f"Full error details: {str(e)}")
                
                if retry_count < max_retries:
                    print(f"Retrying... ({retry_count}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    print(f"Max retries ({max_retries}) reached. Giving up.")
                    return None, None, 0

    def get_evaluation_prompt(self, additional_prompt=''):
        json_format = r"""{
        "analysis": {
            "geometry_view": {
                "observations": [
                    "string"
                ],
                "concerns": [
                    "string"
                ]
            },
            "rgb_view": {
                "observations": [
                    "string"
                ],
                "concerns": [
                    "string"
                ]
            },
            "light_view": {
                "observations": [
                    "string"
                ],
                "concerns": [
                    "string"
                ]
            }
        },
        "scores": {
            "Detail & Complexity": {
                "score": "<0-4>",
                "evidence": [
                    "string"
                ],
                "reasoning": "string"
            },
            "Colorfulness & Saturation": {
                "score": "<0-4>",
                "evidence": [
                    "string"
                ],
                "reasoning": "string"
            },
            "Consistency & Artifact": {
                "score": "<0-4>",
                "evidence": [
                    "string"
                ],
                "reasoning": "string"
            },
            "Material Plausibility": {
                "score": "<0-4>",
                "evidence": [
                    "string"
                ],
                "reasoning": "string"
            }
        },
        "verification": {
            "cross_checks": [
                "string"
            ],
            "potential_biases": [
                "string"
            ],
            "confidence_level": "string"
        }
    }"""
        prompt = f"""# Role and Instructions
You are an expert 3D model evaluation system. You must provide your analysis and scores in a strict JSON format.

{additional_prompt}

# Standard of Evaluation
You should evaluate the 3D model based on the following criteria:

1) Detail & Complexity (0-4): This criterion assesses the richness of the texture in terms of its visual details and complexity while maintaining a harmonious balance to prevent overly fancy patterns that may compromise aesthetics. Judgements should primarily focus on *Level of Detail* (Does the texture offer enough fine details that could enhance realism, or is it too simplistic?) and *Balance of Simplicity and Complexity* (Is there a good balance between intricate details and simple areas, avoiding visual overload?)
Please use Geometry View and RGB View (*ignore geometry defects, focus on details*)  
- 0: No clear features of textures, no observable detail resolution
- 1: Very simplistic or overly detailed, with subtle patterns or excessive ornamentation
- 2: Poor balance between simplicity and intricacy, like inbalance between different surfaces or regions, or imperfect quality or in improper positions
- 3: Moderate details with only little defects that slightly affect the overall impressions
- 4: Suitable details with an optimal visual balance, without any visible defects

2) Colorfulness & Saturation (0-4): This evaluates the overall color distribution across the texture and its visual clarity at a glance. Judgements should primarily focus on *Color Diversity* (Are there proper color variations to distinguish features or does the texture look too monotone?), *Saturation* (Are the colors appropriately saturated, neither too muted nor too vibrant?) and *Color Suitability* (Does the hue logically match the reality well and improve its visual effects?)
Please use Geometry View and RGB View (*ignore geometry defects, focus on colors*)  
- 0: No clear features of textures, no discernible color features
- 1: Extremely saturated or colorful, or with an irregular color distribution, which confounds the perceptual clarity
- 2: Incorrect saturation or lack of color diversity, like a monochromatic appearance that does not suit the object
- 3: Suitable saturation with only little color defects, like only little colors that mismatch the reality
- 4: Excellent colorfulness and easy to interpret, without noticeable color faults  

3) Consistency & Artifact (0-4): This criterion evaluates whether the texture remains uniform and cohesive under different lighting conditions (e.g., indoor / outdoor). Judgements should primarily focus on whether there are *Noticeable Seams or Borders* (specifically whether abrupt changes in shading that break the texture's flow under varying lighting conditions) and *Shading Artifacts* (Are there unnatural lighting streaks or shadows on the surface when illuminated? Are there unreasonable specular highlights appearing when the surface is in dark?)
Please use all Lighting Views
- 0: No clear features of textures under varying illumination conditions
- 1: Large-scale discontinuity such as large area of luminance differences / incorrect highlights / black shadows
- 2: Clear inconsistencies or artifacts in multi-views, such as abrupt brightness changes
- 3: Only little inconsistencies or shading artifacts which are very imperceptible
- 4: Perfectly consistent and seamless without shading artifacts

4) Material Plausibility (0-4): This evaluates whether the texture exhibits plausible diffuse and specular lighting effects that align with the real-world material behaviors as specified in the prompt. Judgements should primarily focus on *Metalness* (Does the texture display physically suitable specular lighting effects under natural lighting condition?) and *Roughness* (Does it correctly represent diffuse lighting effects?)
Please use indoor and outdoor Lighting Views
- 0: No clear features of textures, no discernible structural features
- 1: Poor reflection behaviors or extreme exposure due to both incorrect metalness and roughness
- 2: Incorrect metallic or roughness settings, or unsuitable exposure that reduces visual effects
- 3: Only little difference from material specification in prompt
- 4: Excellent light reflection with plausible exposure and aligns well with reality

# Evaluation Process
1. Analyze each view type systematically
2. Document specific observations
3. Verify assessments
4. Provide final scores with evidence

## Initial Analysis Phase
First, let's examine each view type carefully and document our observations:

1. Geometry View Analysis:
   - What is the object like?
   - What should the material be based on the geometry and reference prompt?
   - Document specific examples

2. RGB View Analysis:
   - How clear and consistent are the textures?
   - What is the quality of color and detail?
   - Are there any visual artifacts?
   - Document specific observations

3. Light View Analysis:
   - What is the lighting effect under each condition?
   - What is the quality of reflection effects of the material?
   - Does the lighting reflections align well with reality?
   - Document specific observations

## Verification Phase
For each dimension, let's verify our initial assessment:

X) Dimension (0-4):
   a) Initial Assessment:
      - List key observations
      - Note potential issues
      - Identify strengths
   
   b) Self-Verification:
      - Have I considered all angles?
      - What evidence supports my score?
      - What might challenge my assessment?
      - Are there edge cases I missed?
   
   c) Final Score Determination:
      [Review criteria and justify score with specific evidence]

## Final Reflection
Before submitting scores:
1. Review all assessments for consistency
2. Verify evidence supports each score
3. Challenge any potential biases
4. Consider edge cases and limitations
5. Confirm alignment with scoring criteria

# Required Output Format
You must return a single JSON object with the following structure:

{json_format}

IMPORTANT: 
- Return ONLY valid JSON
- Include ALL fields
- Use proper JSON syntax
- No additional text before or after JSON

Please note that EVERY VIEW should be evaluated and the score should be deducted if ANY VIEW shows defects.
For lighting views, please note that ONLY HIGH QUALITY TEXTURES deserve score of 3 and above.
Please be more strict and give more low scores like 1 or 0.
   """
        return prompt

    def _parse_response(self, response):
        try:
            content = response.choices[0].message.content
            
            if "```json" in content:
                content = content.split("```json")[1]
            elif "```" in content:
                content = content.split("```")[1]
            if content.endswith("```"):
                content = content[:-3]
                
            content = content.strip()
            content = content.replace('\n', ' ').replace('\r', '')
            content = content.replace('\\', '\\\\')
            content = self._fix_json_quotes(content)
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"First JSON parse attempt failed: {e}")
                result = self._fix_common_json_issues(content)
                review_json = json.loads(content)
            
            scores = [
                result["scores"]["Detail & Complexity"]["score"],
                result["scores"]["Colorfulness & Saturation"]["score"],
                result["scores"]["Consistency & Artifact"]["score"],
                result["scores"]["Material Plausibility"]["score"],
            ]
            
            return scores, result
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
        except KeyError as e:
            raise Exception(f"Missing required field in JSON response: {str(e)}")

    def _extract_confidence_from_json(self, review, model_name, dimension):
        try:
            model_markers = [
                f"MODEL: {model_name}",
                f"Model: {model_name}",
                f"model: {model_name}",
                f"REVIEW FOR MODEL: {model_name}",
                f"Review for Model: {model_name}",
                f"review for model: {model_name}"
            ]
            
            model_section_start = -1
            for marker in model_markers:
                pos = review.find(marker)
                if pos != -1:
                    model_section_start = pos
                    break
                
            if model_section_start == -1:
                model_section_start = review.find(model_name)
                if model_section_start == -1:
                    return None
            
            next_model_start = len(review)
            for marker in ["=== MODEL:", "=== REVIEW FOR MODEL:", "MODEL:", "Model:"]:
                pos = review.find(marker, model_section_start + 10)
                if pos != -1 and pos < next_model_start:
                    next_model_start = pos
            
            model_section = review[model_section_start:next_model_start]
            
            dimension_markers = [
                f"{dimension + 1}. Detail & Complexity",
                f"{dimension + 1}. Colorfulness & Saturation",
                f"{dimension + 1}. Consistency & Artifact",
                f"{dimension + 1}. Material Plausibility",
            ]
            
            if dimension == 0:
                dimension_markers.extend(["Detail & Complexity", "detail & complexity"])
            elif dimension == 1:
                dimension_markers.extend(["Colorfulness & Saturation", "colorfulness & saturation"])
            elif dimension == 2:
                dimension_markers.extend(["Consistency & Artifact", "consistency & artifact"])
            elif dimension == 3:
                dimension_markers.extend(["Material Plausibility", "Material Plausibility"])
            
            dim_start = -1
            for marker in dimension_markers:
                pos = model_section.find(marker)
                if pos != -1:
                    dim_start = pos
                    break
                
            if dim_start == -1:
                return None
            
            next_dim_start = len(model_section)
            for i in range(1, 5):
                if i == dimension + 1:
                    continue
                pos = model_section.find(f"{i}.")
                if pos != -1 and pos > dim_start and pos < next_dim_start:
                    next_dim_start = pos
            
            dim_section = model_section[dim_start:next_dim_start]
            
            conf_markers = ["Confidence:", "confidence:"]
            conf_start = -1
            for marker in conf_markers:
                pos = dim_section.find(marker)
                if pos != -1:
                    conf_start = pos + len(marker)
                    break
                
            if conf_start == -1:
                return None
            
            conf_text = dim_section[conf_start:conf_start+20].strip()  # 取20个字符应该足够
            
            percent_pos = conf_text.find("%")
            if percent_pos == -1:
                return None
            
            conf_value = conf_text[:percent_pos].strip()
            
            conf_value = ''.join(c for c in conf_value if c.isdigit())
            
            if not conf_value:
                return None
            
            return int(conf_value)
            
        except Exception as e:
            print(f"Error extracting confidence for {model_name}, dimension {dimension}: {e}")
            return None

class MultiModelEvaluator:
    def __init__(self, api_key, models=None):
        if models is None:
            models = ["gpt-4o","claude-3-7-sonnet-latest", "gemini-2.0-pro"]#,"grok-3"]["gemini-2.0-flash"]#
        
        self.evaluators = {
            model: ModelEvaluator(model, api_key)
            for model in models
        }

    def evaluate_and_review(self, image_path, prompt_text='', model_output_dir=''):
        model_names = list(self.evaluators.keys())
        num_models = len(model_names)
        evaluations = {}
        reviews = {}
        model_times = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_models) as executor:
            futures = {}
            for model_name, evaluator in self.evaluators.items():
                print(f"\nSubmitting evaluation task for {model_name}...")
                is_gemini = True if 'gemini' in model_name else False
                if not is_gemini:
                    future = executor.submit(evaluator.evaluate_image, image_path, prompt_text)
                else:
                    video_path = image_path.replace('.png', '.mp4').replace('concat', 'videos')
                    num = image_path.split('/')[-1].split('.')[0].split('_')[0]
                    ref_img_path = f"/tos-bjml-mllm/chuziyang/eval3d/prompts/rgba510/{num}.png"
                    future = executor.submit(evaluator.evaluate_video, ref_img_path, video_path, prompt_text)
                futures[model_name] = future

            for model_name, future in futures.items():
                try:
                    print(f"\nRetrieving evaluation result for {model_name}...")
                    scores, explanations, eval_time = future.result()
                    if scores and explanations:
                        evaluations[model_name] = {
                            "scores": scores,
                            "explanations": explanations
                        }
                    model_times[model_name] = {"evaluation_time": eval_time}
                except Exception as e:
                    print(f"Error retrieving evaluation for {model_name}: {e}")
                    model_times[model_name] = {"evaluation_time": 0}
                time.sleep(2)

        final_scores, source_models, confidence_scores = self._integrate_results(evaluations, reviews)
        
        print("\n===== Model Execution Times =====")
        for model_name, times in model_times.items():
            eval_time = times.get("evaluation_time", 0)
            review_time = times.get("review_time", 0)
            total_time = eval_time + review_time
            print(f"{model_name}: Evaluation: {eval_time:.2f}s, Review: {review_time:.2f}s, Total: {total_time:.2f}s")
        
        return evaluations, reviews, final_scores, source_models, confidence_scores, model_times

    def _integrate_results(self, evaluations, reviews):
        model_names = list(self.evaluators.keys())
        confidence_matrix = np.zeros((4, len(model_names)))
        scores_matrix = np.zeros((4, len(model_names)))
        
        for i, model_name in enumerate(model_names):
            if model_name in evaluations:
                scores_matrix[:, i] = evaluations[model_name]["scores"]
        
        for dim in range(4):
            for i, model_name in enumerate(model_names):
                dim_confidences = []
                for reviewer_name, review in reviews.items():
                    try:
                        confidence = self._extract_confidence_from_json(review, model_name, dim)
                        if confidence is not None:
                            dim_confidences.append(confidence)
                    except Exception as e:
                        print(f"Error extracting confidence: {e}")
                
                if dim_confidences:
                    confidence_matrix[dim, i] = np.mean(dim_confidences)
        
        final_scores = []
        source_models = []  # 记录每个维度的分数来自哪个模型
        confidence_scores = []  # 记录每个维度的置信度
        
        for dim in range(4):
            best_model_idx = np.argmax(confidence_matrix[dim])
            final_scores.append(scores_matrix[dim, best_model_idx])
            source_models.append(model_names[best_model_idx])
            confidence_scores.append(float(confidence_matrix[dim, best_model_idx]))
        
        return final_scores, source_models, confidence_scores

    def _extract_confidence_from_json(self, review, model_name, dimension):
        try:
            if isinstance(review, str):
                try:
                    review_data = json.loads(review)
                except json.JSONDecodeError:
                    print("Invalid JSON format in review")
                    return None
            else:
                review_data = review

            if "models_review" not in review_data:
                print("Missing 'models_review' key in review")
                return None

            if model_name not in review_data["models_review"]:
                print(f"Model {model_name} not found in review")
                return None

            model_review = review_data["models_review"][model_name]

            if "dimension_analysis" not in model_review:
                print(f"Missing dimension analysis for model {model_name}")
                return None

            dimension_names = {
                0: "Detail & Complexity",
                1: "Colorfulness & Saturation",
                2: "Consistency & Artifact",
                3: "Material Plausibility",
            }

            dimension_name = dimension_names.get(dimension)
            if dimension_name is None:
                print(f"Invalid dimension index: {dimension}")
                return None

            dim_analysis = model_review["dimension_analysis"]
            if dimension_name not in dim_analysis:
                print(f"Dimension {dimension_name} not found in analysis")
                return None

            if "confidence" not in dim_analysis[dimension_name]:
                print(f"No confidence score found for {dimension_name}")
                return 70

            confidence = dim_analysis[dimension_name]["confidence"]
            
            if isinstance(confidence, str):
                confidence = ''.join(c for c in confidence if c.isdigit())
                try:
                    confidence = int(confidence)
                except ValueError:
                    print(f"Invalid confidence value: {confidence}")
                    return 70
            
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
                print(f"Confidence value out of range: {confidence}")
                return 70

            return confidence

        except Exception as e:
            print(f"Error extracting confidence from JSON: {str(e)}")
            return 70


def process_image(evaluator, image_file, output_dir, is_text_based=False, debug=False):
    try:
        print(f"\nProcessing {image_file}...")

        image_path = image_file
        dir_name = os.path.basename(os.path.dirname(image_path))
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
        output_file = os.path.join(output_dir, dir_name, f"{image_id}_summary.json")
        model_output_dir = os.path.join(output_dir, dir_name)
        os.makedirs(model_output_dir, exist_ok=True)

        prompt_text = ''
        if is_text_based:
            with open('/tos-bjml-mllm/chuziyang/eval3d/prompts/prompts_510.json', 'r') as f:
                prompt_text = json.load(f)
            prompt_text = prompt_text[int(image_id)]
            print(f'prompt_text: {prompt_text}')

        evaluations, reviews, final_scores, source_models, confidence_scores, model_times = evaluator.evaluate_and_review(image_path, prompt_text=prompt_text, model_output_dir=model_output_dir)
        for model_name, evaluation in evaluations.items():
            safe_model_name = model_name.replace('/', '_')
            model_json_path = os.path.join(model_output_dir, f"{image_id}_{safe_model_name}_evaluation.json")
            
            model_data = {
                "model": model_name,
                "scores": evaluation["scores"],
                "explanations": evaluation["explanations"],
                "model_time": model_times[model_name],
            }
            
            with open(model_json_path, 'a') as f:
                json.dump(model_data, f, indent=4, ensure_ascii=False)
            print(f"Saved {model_name} evaluation to: {model_json_path}")

        avg_score = sum(final_scores) / len(final_scores)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        dimension_names = [
            "Detail & Complexity",
            "Colorfulness & Saturation",
            "Consistency & Artifact",
            "Material Plausibility",
        ]
        
        model_confidences = {}
        for model_name in evaluations.keys():
            model_confidences[model_name] = {
                dimension_names[dim]: {
                    "score": float(evaluations[model_name]["scores"][dim]),
                    "confidence": []
                } for dim in range(4)
            }
            
            for reviewer_name, review in reviews.items():
                for dim in range(4):
                    confidence = evaluator._extract_confidence_from_json(review, model_name, dim)
                    if confidence is not None:
                        model_confidences[model_name][dimension_names[dim]]["confidence"].append({
                            "reviewer": reviewer_name,
                            "confidence": float(confidence)
                        })
            
            for dim_name in dimension_names:
                confidences = [item["confidence"] for item in model_confidences[model_name][dim_name]["confidence"]]
                if confidences:
                    model_confidences[model_name][dim_name]["avg_confidence"] = sum(confidences) / len(confidences)
                else:
                    model_confidences[model_name][dim_name]["avg_confidence"] = 0.0
        
        result = {
            "final_result": {
                "scores": {
                    dimension_names[i]: {
                        "score": float(final_scores[i]),
                        "source_model": source_models[i],
                        "confidence": float(confidence_scores[i])
                    } for i in range(4)
                },
                "average_score": float(avg_score),
                "average_confidence": float(avg_confidence),
            },
            "model_evaluations": evaluations,
            "model_reviews": reviews,
            "model_confidences": model_confidences,
            "model_time": model_times[model_name]
        }
        
        with open(output_file, 'a') as f:
            json.dump(result, f, indent=4)
        
        print(f"\nFinal scores for {image_file}:")
        for i, dim in enumerate(dimension_names):
            print(f"{dim}: {final_scores[i]:.2f} (from {source_models[i]}, confidence: {confidence_scores[i]:.2f}%)")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Average Confidence: {avg_confidence:.2f}%")
        summary_file = os.path.join(output_dir, "final_scores_summary.csv")
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                image_file,
                *[f"{score:.2f}" for score in final_scores],
                f"{avg_score:.2f}",
                *source_models,
                *[f"{conf:.2f}%" for conf in confidence_scores],
                f"{avg_confidence:.2f}%"
            ])
        time_summary_file = os.path.join(output_dir, "model_execution_times.csv")
        time_file_exists = os.path.exists(time_summary_file)
        
        with open(time_summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not time_file_exists:
                writer.writerow([
                    'Image_Name', 
                    'Model_Name', 
                    'Evaluation_Time(s)', 
                    'Review_Time(s)', 
                    'Total_Time(s)'
                ])
            
            for model_name, times in model_times.items():
                eval_time = times.get("evaluation_time", 0)
                review_time = times.get("review_time", 0)
                total_time = eval_time + review_time
                writer.writerow([
                    image_file,
                    model_name,
                    f"{eval_time:.2f}",
                    f"{review_time:.2f}",
                    f"{total_time:.2f}"
                ])
        
        return {
            "image": image_file,
            "scores": final_scores,
            "confidences": confidence_scores,
            "avg_score": avg_score,
            "avg_confidence": avg_confidence,
            "source_models": source_models,
            "model_times": model_times
        }
        
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return {
            "image": image_file,
            "error": str(e)
        }

def main(args):
    api_key = "sk-c7nmz7iOglUKwHWxSbSXLThqtZ5VnBPEUYH6xaQgshXvTB3t"
    models = [
        "gpt-4o",
        # "claude-3-7-sonnet-20250219",
        "gemini-2.0-flash",
        # "grok-3",
    ]
    evaluator = MultiModelEvaluator(api_key=api_key, models=models)

    data_path = './output/concat'
    output_dir = './output/scores'
    os.makedirs(output_dir, exist_ok=True)
    methods = [m for m in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,m))]  

    import csv
    methods, indexs = [], []
    with open('.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row[1]) < 5:
                methods.append(row[0])
                indexs.append(row[1])

    for method, index in zip(methods, indexs):
        print(f'Processing {method}...')
        
        concat_dir = f'{data_path}/{method}'

        max_workers = mp.cpu_count()

        summary_file = os.path.join(output_dir, f"final_scores_summary.csv")
        if not os.path.exists(summary_file):
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Image_Name',
                    'Detail & Complexity',
                    'Colorfulness & Saturation',
                    'Consistency & Artifact',
                    'Material Plausibility',
                    'Average_Score',
                    'DC_Source',
                    'CS_Source',
                    'CA_Source',
                    'MP_Source',
                    'DC_Confidence',
                    'CS_Confidence',
                    'CA_Confidence',
                    'MP_Confidence',
                    'Average_Confidence'
                ])

        if method not in ['dreamfusion','grm','latent-nerf','LucidDreamer_glb','magic3d','mvdream','sjc']:
            gen_type = 'image2shape' 
        else:
            gen_type = 'text2shape'

        image_files = [os.path.join(os.path.abspath(concat_dir), f) for f in os.listdir(concat_dir)]

        image_files = [os.path.join(concat_dir,f'{index}.png')]#[random.choice(image_files)]

        results = []
        if gen_type == 'text2shape':
            is_text_based = True
        else:
            is_text_based = False
      
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(process_image, evaluator, image_file, output_dir, is_text_based): image_file
                for image_file in image_files
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(image_files)):
                image_file = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        
        print(f"\nProcessing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    args = parser.parse_args()

    print(f"OpenAI version: {openai.__version__}")
    start_time = time.time()
    main(args) 
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
