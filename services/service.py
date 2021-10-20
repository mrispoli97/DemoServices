from actions.actions import Action
from utility.redis import RequestPipeline, ResponseTable
from pprint import pprint

action = Action()


class Service:

    def __init__(self, data):
        self._data = data

    def run(self):
        pass


class ClassificationService(Service):

    def __init__(self, data):
        super(ClassificationService, self).__init__(data)

    def run(self):
        global action
        model = self._data['model']
        filepath = self._data['filepath']
        prediction = action.classify(model=model, filepath=filepath)
        return prediction


class ObfuscationService(Service):
    def __init__(self, data):
        super(ObfuscationService, self).__init__(data)

    def run(self):
        global action
        obfuscation = self._data['obfuscation']
        severity = self._data['severity']
        filepath = self._data['filepath']
        dst = self._data['dst']
        obfuscated_filepath = action.obfuscate(filepath=filepath, dst=dst, config={
            "obfuscation": obfuscation,
            "params": {
                "severity": float(severity)
            }
        })
        return obfuscated_filepath


class ChallengesManager:

    def __init__(self):
        self._pipeline = RequestPipeline()
        self._table = ResponseTable()

    def run(self):
        print("Starting...")
        while True:
            if len(self._pipeline) > 0:
                request = self._pipeline.pop()
                job = request['job']
                if job == 'classification':
                    cs = ClassificationService(request['data'])
                    prediction = cs.run()
                    self._table.set(request['id'], prediction)
                    print(f"CLASSIFICATION: {prediction}")
                elif job == 'obfuscation':
                    obs = ObfuscationService(request['data'])
                    ob_filepath = obs.run()
                    self._table.set(request['id'], ob_filepath)
                    print(f"OBFUSCATION: {ob_filepath}")


def main():
    cm = ChallengesManager()
    cm.run()


if __name__ == '__main__':
    main()
