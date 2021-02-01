from pathlib import Path

from sacred.observers import FileStorageObserver, SlackObserver


def add_file_storage_observer(experiment):
    file_observer_dir = Path(__file__).parent / 'runs'
    file_observer_dir.mkdir(exist_ok=True)
    file_observer = FileStorageObserver(file_observer_dir)
    experiment.observers.append(file_observer)
    return file_observer


def add_slack_observer(experiment):
    slack_config_path = Path('~/slack.json').expanduser()
    if not slack_config_path.is_file():
        print('Slack config path not found:', slack_config_path)
        slack_observer = None
    else:
        slack_observer = SlackObserver.from_config(str(slack_config_path))
        experiment.observers.append(slack_observer)
    return slack_observer


def add_observers(experiment):
    add_file_storage_observer(experiment)
    add_slack_observer(experiment)
