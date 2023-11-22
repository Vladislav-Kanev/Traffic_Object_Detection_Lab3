import pandas as pd
from tqdm import tqdm

from src.trainer import SubmissionType


def prepare_result(results: SubmissionType, result_path: str, index_map : list[int]) -> None:
    submission_result = []
    global_id = 0
    for result_id, sample_result in enumerate(tqdm(results), desc=f'Processing'):
        image_id = index_map[result_id]
        
        for bbox, category_id, score in zip(sample_result['boxes'], sample_result['labels'],
                                            sample_result['scores']):
            submission_result.append(
                {
                    'id': global_id,
                    'image_id': image_id,
                    'category_id': category_id.cpu.item(),
                    'bbox': bbox.cpu.tolist(),
                    'score': score.cpu.item()
                }
            )
            global_id += 1
    submission_results_pd = pd.DataFrame(submission_result)

    with open(result_path, mode='w') as file_descr:
        submission_results_pd.to_csv(file_descr, index=False)
    print(f'Submission file is saved as {result_path}', flush=True)
