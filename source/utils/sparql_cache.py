import json
import os.path
from pathlib import Path
from threading import Thread
from time import sleep

from utils.sparql_executer import is_reachable, is_reachable_cmp, is_intersectant, get_types, get_in_relations, \
    get_out_relations, \
    get_in_entities, \
    get_out_entities, \
    get_entities_cmp, get_in_entities_for_literal, get_in_attributes

path = str(Path(__file__).parent.absolute())


class SparqlCache:
    def __init__(self, dataset: str = "graphq"):
        self.dataset = dataset
        if dataset == "grail":
            if os.path.exists(path + "/../cache/execution.json"):
                with open(path + "/../cache/execution.json") as f:
                    self.execution = json.load(f)
                    if "is_reachable" not in self.execution:
                        self.execution["is_reachable"] = {}
                    if "is_intersectant" not in self.execution:
                        self.execution["is_intersectant"] = {}
            else:
                self.execution = {"types": {}, "in_relations": {}, "out_relations": {}, "in_entities": {},
                                      "out_entities": {},
                                      "cmp_entities": {},
                                      "is_reachable": {},
                                      "is_intersectant": {}}
        elif dataset == "graphq":
            if os.path.exists(path + "/../cache/gq1_execution.json"):
                with open(path + "/../cache/gq1_execution.json") as f:
                    self.execution = json.load(f)
                    if "is_reachable" not in self.execution:
                        self.execution["is_reachable"] = {}
                    if "is_intersectant" not in self.execution:
                        self.execution["is_intersectant"] = {}
            else:
                self.execution = {"types": {}, "in_relations": {}, "out_relations": {}, "in_entities": {},
                                      "out_entities": {},
                                      "cmp_entities": {},
                                      "is_reachable": {},
                                      "is_intersectant": {}}
        elif dataset == "webq":
            if os.path.exists(path + "/../cache/webq_execution.json"):
                with open(path + "/../cache/webq_execution.json") as f:
                    self.execution = json.load(f)
                    if "is_reachable" not in self.execution:
                        self.execution["is_reachable"] = {}
                    if "is_intersectant" not in self.execution:
                        self.execution["is_intersectant"] = {}
            else:
                self.execution = {"types": {}, "in_relations": {}, "out_relations": {}, "in_entities": {},
                                      "out_entities": {},
                                      "cmp_entities": {},
                                      "is_reachable": {},
                                      "is_intersectant": {}}


    def is_intersectant(self, derivation1, derivation2):
        derivation1[1] = tuple(derivation1[1])
        derivation1 = tuple(derivation1)
        derivation2[1] = tuple(derivation2[1])
        derivation2 = tuple(derivation2)

        if derivation2[0] < derivation1[0]:
            tmp = derivation1
            derivation1 = derivation2
            derivation2 = tmp

        if str((derivation1, derivation2)) not in self.execution["is_intersectant"]:
            self.execution["is_intersectant"][str((derivation1, derivation2))] = is_intersectant(derivation1,
                                                                                                     derivation2)

        return self.execution["is_intersectant"][str((derivation1, derivation2))]

    def is_reachable(self, derivation, answer_type):
        # derivation = (start, [property1,...])

        derivation[1] = tuple(derivation[1])
        derivation = tuple(derivation)

        if str((derivation, answer_type)) not in self.execution["is_reachable"]:
            if len(derivation) == 2:
                self.execution["is_reachable"][str((derivation, answer_type))] = is_reachable(derivation,
                                                                                                  answer_type)
            elif len(derivation) == 3:
                self.execution["is_reachable"][str((derivation, answer_type))] = is_reachable_cmp(derivation,
                                                                                                      answer_type)

        return self.execution["is_reachable"][str((derivation, answer_type))]

    def get_types(self, entity):
        if entity not in self.execution["types"]:
            # print("not hit")
            self.execution["types"][entity] = get_types(entity)
            sleep(0.02)

        return self.execution["types"][entity]

    def get_in_relations(self, entity):
        if entity not in self.execution["in_relations"]:
            # print("not hit")
            self.execution["in_relations"][entity] = list(get_in_relations(entity))
            sleep(0.02)

        return set(self.execution["in_relations"][entity])

    def get_in_attributes(self, literal):
        if literal not in self.execution["in_relations"]:
            self.execution["in_relations"][literal] = list(get_in_attributes(literal))
            sleep(0.02)

        return set(self.execution["in_relations"][literal])

    def get_out_relations(self, entity):
        if entity not in self.execution["out_relations"]:
            # print("not hit")
            self.execution["out_relations"][entity] = list(get_out_relations(entity))
            sleep(0.02)

        return set(self.execution["out_relations"][entity])

    def get_in_entities(self, entity, relation):
        if entity + relation not in self.execution["in_entities"]:
            self.execution["in_entities"][entity + relation] = list(get_in_entities(entity, relation))
            sleep(0.02)

        return set(self.execution["in_entities"][entity + relation])

    def get_in_entities_for_literal(self, value, relation):
        if value + relation not in self.execution["in_entities"]:
            self.execution["in_entities"][value + relation] = list(get_in_entities_for_literal(value, relation))
            sleep(0.02)

        return set(self.execution["in_entities"][value + relation])

    def get_out_entities(self, entity, relation):
        if entity + relation not in self.execution["out_entities"]:
            self.execution["out_entities"][entity + relation] = list(get_out_entities(entity, relation))
            sleep(0.02)

        return set(self.execution["out_entities"][entity + relation])

    def get_entities_cmp(self, value, relation, comp):
        if str(value) + relation + comp not in self.execution["cmp_entities"]:
            self.execution["cmp_entities"][str(value) + relation + comp] = list(
                get_entities_cmp(value, relation, comp))
            sleep(0.02)

        return self.execution["cmp_entities"][str(value) + relation + comp]

    def cache_results(self):
        if self.dataset == "grail":
            with open(path + "/../cache/execution.json", 'w') as f1:
                json.dump(self.execution, f1)
        elif self.dataset == "gq1":
            with open(path + "/../cache/gq1_execution.json", 'w') as f1:
                json.dump(self.execution, f1)
        elif self.dataset == "webq":
            with open(path + "/../cache/webq_execution.json", 'w') as f1:
                json.dump(self.execution, f1)
        elif self.dataset == "cwq":
            with open(path + "/../cache/cwq_execution.json", 'w') as f1:
                json.dump(self.execution, f1)


if __name__ == '__main__':
    sparql_cache = SparqlCache()
    print(sparql_cache.get_out_entities('m.01gbbz', "film.actor.film"))
