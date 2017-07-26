import os
import sys

import constants
import evala
import evala_generator
import evalb
import evalb_generator
import model_evaluation
import utils


def evaluate_pipe(evaluation_dir, gold_dir, task, model, config):
    if not os.path.exists(constants.TEMP_OUTPUT):
        os.makedirs(constants.TEMP_OUTPUT)
    if task == "A":
        evala_generator.generate(evaluation_dir, constants.TEMP_OUTPUT, model, config)
        evala.evaluate(constants.TEMP_OUTPUT, gold_dir)
    else:
        evalb_generator.generate(evaluation_dir, constants.TEMP_OUTPUT, model, config)
        evalb.evaluate(constants.TEMP_OUTPUT, gold_dir)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage:', __file__, '<evaluation_dir> <gold_dir> <task>')
        print(
            'evaluation_dir directory contains evaluation tsv files for each theme.')
        print('gold_dir directory contains gold tsv files for each theme.')
        print('<task> represent evaluation task: "A" or "B"')
        print('Example call:')
        print(
            'python3 evaluator.py ../dataset/evaluation_data ../dataset/gold_data A '
            'model_path config_path')
        sys.exit(1)

    _, evaluation_dir, gold_dir, task, model_path, config_path = sys.argv

    model = model_evaluation.ModelEvaluator(model_path)
    config = utils.read_config(config_path)

    evaluate_pipe(evaluation_dir, gold_dir, task, model, config)
