# Evaluation and Safety

Master comprehensive evaluation methodologies and safety measures for transformer models, from traditional metrics to modern LLM evaluation techniques and responsible AI practices.

## 🎯 Learning Objectives

By the end of this module, you will understand:
- Traditional and modern evaluation metrics
- Benchmark design and implementation
- Human evaluation methodologies
- Safety and bias assessment
- Robustness testing
- Hallucination detection and mitigation
- Responsible AI practices

## 📚 Table of Contents

1. [Evaluation Fundamentals](#1-evaluation-fundamentals)
2. [Traditional Metrics](#2-traditional-metrics)
3. [LLM-Specific Evaluation](#3-llm-specific-evaluation)
4. [Benchmark Suites](#4-benchmark-suites)
5. [Human Evaluation](#5-human-evaluation)
6. [Safety Assessment](#6-safety-assessment)
7. [Bias Detection and Mitigation](#7-bias-detection-and-mitigation)
8. [Robustness Testing](#8-robustness-testing)

## 1. Evaluation Fundamentals

### 1.1 Why Evaluation Matters

Comprehensive evaluation is crucial for:
- **Performance Assessment**: Understanding model capabilities
- **Comparison**: Benchmarking against other models
- **Safety Verification**: Ensuring responsible deployment
- **Improvement Guidance**: Identifying weaknesses
- **Trust Building**: Providing transparency

### 1.2 Evaluation Challenges

Modern LLMs present unique evaluation challenges:

| Challenge | Description | Solution Approach |
|-----------|-------------|-------------------|
| **Multitask Nature** | Models perform many tasks | Task-specific evaluation |
| **Subjective Quality** | Many outputs have no single correct answer | Human evaluation + multiple metrics |
| **Emergent Abilities** | Capabilities appear at scale | Continuous evaluation |
| **Safety Concerns** | Potential for harmful outputs | Red teaming + safety benchmarks |
| **Computational Cost** | Large models expensive to evaluate | Efficient evaluation strategies |

### 1.3 Evaluation Framework

```python
class EvaluationFramework:
    """Comprehensive evaluation framework."""
    
    def __init__(self):
        self.metrics = {}
        self.benchmarks = {}
        self.safety_checks = {}
        
    def evaluate(self, model, test_data):
        results = {
            'performance': self.evaluate_performance(model, test_data),
            'safety': self.evaluate_safety(model, test_data),
            'robustness': self.evaluate_robustness(model, test_data),
            'bias': self.evaluate_bias(model, test_data)
        }
        return results
```

## 2. Traditional Metrics

### 2.1 Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_classification(predictions, labels):
    """Evaluate classification performance."""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

### 2.2 Generation Metrics

**BLEU (Bilingual Evaluation Understudy)**
```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    """Calculate BLEU score."""
    reference = [reference.split()]
    candidate = candidate.split()
    
    # Calculate n-gram scores
    bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    
    return {
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_4': bleu_4
    }
```

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
```python
from rouge import Rouge

def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores."""
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    
    return {
        'rouge_1': scores['rouge-1']['f'],
        'rouge_2': scores['rouge-2']['f'],
        'rouge_l': scores['rouge-l']['f']
    }
```

### 2.3 Perplexity

```python
import torch
import torch.nn.functional as F

def calculate_perplexity(model, test_data):
    """Calculate model perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_data:
            outputs = model(batch['input_ids'])
            loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                batch['labels'].view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += (batch['labels'] != -100).sum().item()
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()
```

## 3. LLM-Specific Evaluation

### 3.1 Task-Specific Evaluation

```python
class TaskEvaluator:
    """Evaluate specific tasks."""
    
    def evaluate_question_answering(self, model, qa_dataset):
        """Evaluate QA performance."""
        correct = 0
        total = 0
        
        for example in qa_dataset:
            prompt = f"Question: {example['question']}\nAnswer:"
            generated = model.generate(prompt, max_length=50)
            
            # Exact match
            if example['answer'].lower() in generated.lower():
                correct += 1
            total += 1
            
        return {'exact_match': correct / total}
    
    def evaluate_summarization(self, model, summary_dataset):
        """Evaluate summarization quality."""
        rouge_scores = []
        
        for example in summary_dataset:
            prompt = f"Summarize: {example['document']}\nSummary:"
            generated = model.generate(prompt, max_length=100)
            
            scores = calculate_rouge(example['summary'], generated)
            rouge_scores.append(scores)
            
        # Average scores
        avg_scores = {}
        for metric in rouge_scores[0].keys():
            avg_scores[metric] = sum(s[metric] for s in rouge_scores) / len(rouge_scores)
            
        return avg_scores
```

### 3.2 Instruction Following

```python
class InstructionEvaluator:
    """Evaluate instruction-following capabilities."""
    
    def evaluate_instruction_following(self, model, instruction_dataset):
        """Test instruction adherence."""
        results = []
        
        for example in instruction_dataset:
            response = model.generate(example['instruction'])
            
            # Check constraints
            score = self.check_constraints(
                response,
                example.get('constraints', {})
            )
            results.append(score)
            
        return {
            'instruction_score': sum(results) / len(results),
            'perfect_following': sum(r == 1.0 for r in results) / len(results)
        }
    
    def check_constraints(self, response, constraints):
        """Check if response meets constraints."""
        score = 1.0
        
        if 'max_length' in constraints:
            if len(response.split()) > constraints['max_length']:
                score *= 0.8
                
        if 'format' in constraints:
            if not self.check_format(response, constraints['format']):
                score *= 0.5
                
        if 'must_include' in constraints:
            for term in constraints['must_include']:
                if term not in response:
                    score *= 0.7
                    
        return score
```

### 3.3 Reasoning Evaluation

```python
class ReasoningEvaluator:
    """Evaluate reasoning capabilities."""
    
    def evaluate_chain_of_thought(self, model, reasoning_dataset):
        """Evaluate chain-of-thought reasoning."""
        results = {
            'correct_answer': 0,
            'valid_reasoning': 0,
            'step_accuracy': []
        }
        
        for example in reasoning_dataset:
            # Generate with CoT prompt
            prompt = f"{example['question']}\nLet's think step by step:"
            response = model.generate(prompt, max_length=200)
            
            # Parse reasoning steps
            steps = self.parse_reasoning_steps(response)
            
            # Check final answer
            if example['answer'] in response:
                results['correct_answer'] += 1
                
            # Validate reasoning
            valid_steps = self.validate_reasoning(steps, example)
            results['step_accuracy'].append(valid_steps)
            
            if valid_steps > 0.8:
                results['valid_reasoning'] += 1
                
        n = len(reasoning_dataset)
        return {
            'answer_accuracy': results['correct_answer'] / n,
            'reasoning_validity': results['valid_reasoning'] / n,
            'avg_step_accuracy': sum(results['step_accuracy']) / n
        }
```

## 4. Benchmark Suites

### 4.1 Popular Benchmarks

| Benchmark | Focus Area | Tasks | Description |
|-----------|------------|-------|-------------|
| **GLUE** | Language Understanding | 9 tasks | Classification and similarity |
| **SuperGLUE** | Advanced Understanding | 8 tasks | More challenging than GLUE |
| **MMLU** | Knowledge | 57 subjects | Multiple-choice questions |
| **Big Bench** | Diverse Capabilities | 200+ tasks | Comprehensive evaluation |
| **HELM** | Holistic Evaluation | 42 scenarios | Multi-metric evaluation |
| **HumanEval** | Code Generation | 164 problems | Programming challenges |

### 4.2 Benchmark Implementation

```python
class BenchmarkSuite:
    """Run comprehensive benchmarks."""
    
    def __init__(self):
        self.benchmarks = {
            'mmlu': self.run_mmlu,
            'humaneval': self.run_humaneval,
            'truthfulqa': self.run_truthfulqa
        }
        
    def run_mmlu(self, model):
        """Run MMLU benchmark."""
        subjects = load_mmlu_subjects()
        results = {}
        
        for subject in subjects:
            correct = 0
            total = 0
            
            for question in subject['questions']:
                # Format as multiple choice
                prompt = self.format_mmlu_prompt(question)
                response = model.generate(prompt, max_length=1)
                
                if response.strip().upper() == question['answer']:
                    correct += 1
                total += 1
                
            results[subject['name']] = correct / total
            
        return {
            'average': sum(results.values()) / len(results),
            'per_subject': results
        }
    
    def run_humaneval(self, model):
        """Run HumanEval code generation benchmark."""
        problems = load_humaneval_problems()
        passed = 0
        
        for problem in problems:
            # Generate code
            prompt = problem['prompt']
            generated_code = model.generate(prompt, max_length=200)
            
            # Test generated code
            try:
                exec(generated_code + '\n' + problem['test_code'])
                passed += 1
            except:
                pass
                
        return {'pass_rate': passed / len(problems)}
```

## 5. Human Evaluation

### 5.1 Human Evaluation Framework

```python
class HumanEvaluationFramework:
    """Framework for human evaluation."""
    
    def __init__(self):
        self.criteria = {
            'fluency': 'How natural and grammatically correct is the text?',
            'coherence': 'How well do ideas flow and connect?',
            'relevance': 'How well does the response address the prompt?',
            'informativeness': 'How informative and useful is the response?',
            'safety': 'Is the response safe and appropriate?'
        }
        
    def prepare_evaluation_batch(self, model_outputs, reference_outputs=None):
        """Prepare batch for human evaluation."""
        batch = []
        
        for i, output in enumerate(model_outputs):
            item = {
                'id': i,
                'prompt': output['prompt'],
                'response': output['response'],
                'criteria': self.criteria
            }
            
            if reference_outputs:
                item['reference'] = reference_outputs[i]
                
            batch.append(item)
            
        return batch
    
    def calculate_inter_rater_agreement(self, ratings):
        """Calculate agreement between raters."""
        from sklearn.metrics import cohen_kappa_score
        
        # Assuming ratings is dict of rater_id -> list of scores
        raters = list(ratings.keys())
        
        if len(raters) < 2:
            return None
            
        # Pairwise kappa scores
        kappa_scores = []
        
        for i in range(len(raters)):
            for j in range(i + 1, len(raters)):
                kappa = cohen_kappa_score(
                    ratings[raters[i]],
                    ratings[raters[j]]
                )
                kappa_scores.append(kappa)
                
        return {
            'average_kappa': sum(kappa_scores) / len(kappa_scores),
            'min_kappa': min(kappa_scores),
            'max_kappa': max(kappa_scores)
        }
```

### 5.2 A/B Testing

```python
class ABTestFramework:
    """A/B testing for model comparison."""
    
    def run_preference_test(self, model_a, model_b, test_prompts):
        """Run preference test between two models."""
        preferences = {
            'model_a': 0,
            'model_b': 0,
            'tie': 0
        }
        
        detailed_results = []
        
        for prompt in test_prompts:
            # Generate from both models
            response_a = model_a.generate(prompt)
            response_b = model_b.generate(prompt)
            
            # Random order for blind evaluation
            if random.random() > 0.5:
                first, second = response_a, response_b
                order = ['a', 'b']
            else:
                first, second = response_b, response_a
                order = ['b', 'a']
                
            # Get preference (simulated here)
            preference = self.get_human_preference(
                prompt, first, second
            )
            
            # Record result
            if preference == 'first':
                preferences[f'model_{order[0]}'] += 1
            elif preference == 'second':
                preferences[f'model_{order[1]}'] += 1
            else:
                preferences['tie'] += 1
                
            detailed_results.append({
                'prompt': prompt,
                'preference': preference,
                'order': order
            })
            
        return preferences, detailed_results
```

## 6. Safety Assessment

### 6.1 Safety Evaluation Framework

```python
class SafetyEvaluator:
    """Comprehensive safety evaluation."""
    
    def __init__(self):
        self.safety_categories = [
            'violence',
            'hate_speech',
            'sexual_content',
            'self_harm',
            'illegal_activity',
            'personal_info',
            'misinformation'
        ]
        
    def evaluate_safety(self, model, safety_prompts):
        """Evaluate model safety."""
        results = {
            'safe_responses': 0,
            'unsafe_responses': 0,
            'refusals': 0,
            'by_category': {cat: 0 for cat in self.safety_categories}
        }
        
        for prompt_data in safety_prompts:
            prompt = prompt_data['prompt']
            category = prompt_data['category']
            expected = prompt_data['expected_behavior']
            
            response = model.generate(prompt)
            
            # Check response
            safety_score = self.score_safety(response, category)
            
            if safety_score['is_safe']:
                results['safe_responses'] += 1
            else:
                results['unsafe_responses'] += 1
                results['by_category'][category] += 1
                
            if safety_score['is_refusal']:
                results['refusals'] += 1
                
        total = len(safety_prompts)
        results['safety_rate'] = results['safe_responses'] / total
        results['refusal_rate'] = results['refusals'] / total
        
        return results
    
    def score_safety(self, response, category):
        """Score safety of response."""
        # Check for refusal patterns
        refusal_patterns = [
            "I cannot", "I won't", "I'm not able to",
            "I don't feel comfortable", "inappropriate"
        ]
        
        is_refusal = any(pattern in response for pattern in refusal_patterns)
        
        # Check for unsafe content (simplified)
        unsafe_keywords = self.get_unsafe_keywords(category)
        is_unsafe = any(keyword in response.lower() for keyword in unsafe_keywords)
        
        return {
            'is_safe': not is_unsafe or is_refusal,
            'is_refusal': is_refusal,
            'category': category
        }
```

### 6.2 Red Teaming

```python
class RedTeamEvaluator:
    """Red team testing for adversarial evaluation."""
    
    def __init__(self):
        self.attack_types = [
            'prompt_injection',
            'jailbreaking',
            'data_extraction',
            'harmful_content',
            'bias_amplification'
        ]
        
    def run_red_team_evaluation(self, model, attack_suite):
        """Run red team attacks."""
        results = {
            'total_attacks': len(attack_suite),
            'successful_defenses': 0,
            'vulnerabilities': [],
            'by_attack_type': {t: {'attempts': 0, 'defended': 0} 
                              for t in self.attack_types}
        }
        
        for attack in attack_suite:
            attack_type = attack['type']
            prompt = attack['prompt']
            success_criteria = attack['success_criteria']
            
            # Run attack
            response = model.generate(prompt)
            
            # Check if attack succeeded
            attack_succeeded = self.check_attack_success(
                response, success_criteria
            )
            
            results['by_attack_type'][attack_type]['attempts'] += 1
            
            if not attack_succeeded:
                results['successful_defenses'] += 1
                results['by_attack_type'][attack_type]['defended'] += 1
            else:
                results['vulnerabilities'].append({
                    'type': attack_type,
                    'prompt': prompt,
                    'response': response
                })
                
        results['defense_rate'] = results['successful_defenses'] / results['total_attacks']
        
        return results
```

## 7. Bias Detection and Mitigation

### 7.1 Bias Detection

```python
class BiasDetector:
    """Detect various forms of bias."""
    
    def __init__(self):
        self.bias_categories = [
            'gender',
            'race',
            'religion',
            'nationality',
            'age',
            'socioeconomic'
        ]
        
    def detect_demographic_bias(self, model, bias_test_set):
        """Detect demographic biases."""
        results = {cat: {'bias_score': 0, 'examples': []} 
                  for cat in self.bias_categories}
        
        for test in bias_test_set:
            category = test['category']
            templates = test['templates']
            attributes = test['attributes']
            
            # Test each demographic pair
            for template in templates:
                scores = []
                
                for demo_a, demo_b in test['demographic_pairs']:
                    # Generate with both demographics
                    prompt_a = template.format(demographic=demo_a)
                    prompt_b = template.format(demographic=demo_b)
                    
                    response_a = model.generate(prompt_a)
                    response_b = model.generate(prompt_b)
                    
                    # Measure bias
                    bias_score = self.measure_response_bias(
                        response_a, response_b, attributes
                    )
                    scores.append(bias_score)
                    
                    if abs(bias_score) > 0.2:  # Significant bias
                        results[category]['examples'].append({
                            'template': template,
                            'demo_a': demo_a,
                            'demo_b': demo_b,
                            'bias_score': bias_score
                        })
                        
                results[category]['bias_score'] = sum(scores) / len(scores)
                
        return results
    
    def measure_response_bias(self, response_a, response_b, attributes):
        """Measure bias between two responses."""
        # Count attribute occurrences
        score_a = sum(attr in response_a.lower() for attr in attributes)
        score_b = sum(attr in response_b.lower() for attr in attributes)
        
        # Normalize by response length
        score_a /= max(len(response_a.split()), 1)
        score_b /= max(len(response_b.split()), 1)
        
        # Return bias score (positive means bias toward A)
        return score_a - score_b
```

### 7.2 Fairness Metrics

```python
class FairnessEvaluator:
    """Evaluate model fairness."""
    
    def evaluate_fairness(self, model, fairness_dataset):
        """Comprehensive fairness evaluation."""
        metrics = {
            'demographic_parity': self.compute_demographic_parity(model, fairness_dataset),
            'equal_opportunity': self.compute_equal_opportunity(model, fairness_dataset),
            'calibration': self.compute_calibration(model, fairness_dataset)
        }
        
        return metrics
    
    def compute_demographic_parity(self, model, dataset):
        """Compute demographic parity difference."""
        group_rates = {}
        
        for group in dataset.get_demographic_groups():
            group_data = dataset.filter_by_group(group)
            predictions = model.predict(group_data)
            
            positive_rate = sum(predictions) / len(predictions)
            group_rates[group] = positive_rate
            
        # Calculate maximum difference
        rates = list(group_rates.values())
        dp_difference = max(rates) - min(rates)
        
        return {
            'dp_difference': dp_difference,
            'group_rates': group_rates,
            'fair': dp_difference < 0.1  # Threshold
        }
```

## 8. Robustness Testing

### 8.1 Adversarial Robustness

```python
class RobustnessEvaluator:
    """Test model robustness."""
    
    def evaluate_robustness(self, model, robustness_suite):
        """Comprehensive robustness evaluation."""
        results = {
            'perturbation_robustness': self.test_perturbations(model, robustness_suite),
            'consistency': self.test_consistency(model, robustness_suite),
            'out_of_distribution': self.test_ood(model, robustness_suite)
        }
        
        return results
    
    def test_perturbations(self, model, test_set):
        """Test robustness to input perturbations."""
        perturbation_types = [
            'typos',
            'synonyms',
            'paraphrases',
            'case_changes',
            'punctuation'
        ]
        
        results = {}
        
        for pert_type in perturbation_types:
            correct_original = 0
            correct_perturbed = 0
            consistency = 0
            
            for example in test_set:
                original = example['text']
                label = example['label']
                
                # Get original prediction
                pred_original = model.predict(original)
                if pred_original == label:
                    correct_original += 1
                    
                # Generate perturbation
                perturbed = self.apply_perturbation(original, pert_type)
                
                # Get perturbed prediction
                pred_perturbed = model.predict(perturbed)
                if pred_perturbed == label:
                    correct_perturbed += 1
                    
                if pred_original == pred_perturbed:
                    consistency += 1
                    
            n = len(test_set)
            results[pert_type] = {
                'original_accuracy': correct_original / n,
                'perturbed_accuracy': correct_perturbed / n,
                'consistency_rate': consistency / n,
                'robustness_score': correct_perturbed / max(correct_original, 1)
            }
            
        return results
```

### 8.2 Hallucination Detection

```python
class HallucinationDetector:
    """Detect and measure hallucinations."""
    
    def detect_hallucinations(self, model, factual_dataset):
        """Detect factual hallucinations."""
        results = {
            'hallucination_rate': 0,
            'types': {
                'entity': 0,
                'relation': 0,
                'attribute': 0,
                'temporal': 0
            },
            'examples': []
        }
        
        total_claims = 0
        hallucinated_claims = 0
        
        for example in factual_dataset:
            prompt = example['prompt']
            facts = example['facts']
            
            # Generate response
            response = model.generate(prompt)
            
            # Extract claims from response
            claims = self.extract_claims(response)
            total_claims += len(claims)
            
            # Check each claim
            for claim in claims:
                claim_type = self.classify_claim(claim)
                is_hallucination = self.verify_claim(claim, facts)
                
                if is_hallucination:
                    hallucinated_claims += 1
                    results['types'][claim_type] += 1
                    
                    if len(results['examples']) < 10:
                        results['examples'].append({
                            'prompt': prompt,
                            'claim': claim,
                            'type': claim_type,
                            'response': response
                        })
                        
        results['hallucination_rate'] = hallucinated_claims / max(total_claims, 1)
        
        return results
```

## 📊 Evaluation Best Practices

### Comprehensive Evaluation Strategy

```python
class ComprehensiveEvaluator:
    """Complete evaluation pipeline."""
    
    def __init__(self):
        self.evaluators = {
            'performance': PerformanceEvaluator(),
            'safety': SafetyEvaluator(),
            'bias': BiasDetector(),
            'robustness': RobustnessEvaluator(),
            'human': HumanEvaluationFramework()
        }
        
    def full_evaluation(self, model, eval_suite):
        """Run complete evaluation."""
        results = {}
        
        # Automated evaluations
        for name, evaluator in self.evaluators.items():
            if name != 'human':
                results[name] = evaluator.evaluate(model, eval_suite[name])
                
        # Human evaluation on subset
        human_subset = self.select_human_eval_subset(results)
        results['human'] = self.evaluators['human'].evaluate(
            model, human_subset
        )
        
        # Generate report
        report = self.generate_evaluation_report(results)
        
        return results, report
```

## 📈 Evaluation Metrics Summary

| Category | Metrics | When to Use |
|----------|---------|-------------|
| **Performance** | Accuracy, F1, Perplexity | Basic capability assessment |
| **Generation** | BLEU, ROUGE, BERTScore | Text generation quality |
| **Safety** | Toxicity, Bias scores | Pre-deployment checks |
| **Robustness** | Perturbation accuracy | Reliability testing |
| **Human** | Preference, Quality ratings | Final validation |

## 🛡️ Safety Checklist

Before deployment, ensure:

- [ ] Safety evaluation on standard benchmarks
- [ ] Red team testing completed
- [ ] Bias assessment across demographics
- [ ] Robustness to adversarial inputs
- [ ] Hallucination rate acceptable
- [ ] Human evaluation satisfactory
- [ ] Monitoring plan in place
- [ ] Incident response prepared

## 📝 Summary

Comprehensive evaluation requires:
- **Multiple Perspectives**: Automated and human evaluation
- **Safety First**: Rigorous safety and bias testing
- **Continuous Process**: Ongoing evaluation post-deployment
- **Transparency**: Clear reporting of limitations
- **Responsible Practices**: Ethical considerations throughout

## ➡️ Next Steps

Ready to apply transformers in practice? Head to [Topic 15: Real-world Applications](../15-real-world-applications/) to explore practical implementations and deployment strategies!