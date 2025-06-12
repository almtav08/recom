from collections import Counter, defaultdict, deque
import random


class MarkovPathGenerator:

    def __init__(self, order=2, final_exam_id=None, max_repeat=3):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.start_tokens = []
        self.final_exam_id = final_exam_id
        self.transition_counts = Counter()
        self.max_repeat = max_repeat

    def train(self, paths):
        """
        paths: Lista de listas. Cada lista es un path de recursos (e.g. [1, 2, 3])
        """
        for path in paths:
            if len(path) <= self.order:
                continue

            self.start_tokens.append(tuple(path[: self.order]))
            window = deque(maxlen=self.order)

            for i, item in enumerate(path):
                if self.final_exam_id is not None and item == self.final_exam_id:
                    window.clear()
                    continue

                window.append(item)
                if len(window) == self.order and i + 1 < len(path):
                    next_item = path[i + 1]
                    if (
                        self.final_exam_id is not None
                        and next_item == self.final_exam_id
                    ):
                        continue
                    context = tuple(window)
                    self.transitions[context][next_item] += 1
                    self.transition_counts[(context, next_item)] += 1

    def _weighted_choice(self, counter):
        """
        Selecciona aleatoriamente un elemento según su peso (frecuencia).
        """
        total = sum(counter.values())
        r = random.uniform(0, total)
        upto = 0
        for key, count in counter.items():
            if upto + count >= r:
                return key
            upto += count
        # Fallback por si acaso
        return random.choice(list(counter.keys()))

    def generate_path(self, max_length=10):
        if not self.start_tokens:
            raise ValueError("No se han entrenado paths.")

        current = list(random.choice(self.start_tokens))
        path = current[:]
        repeat_counter = Counter()

        while len(path) < max_length:
            context = tuple(path[-self.order:])
            if context not in self.transitions:
                break

            choices = self.transitions[context]
            next_item = self._weighted_choice(choices)

            # Evitar repeticiones excesivas
            repeat_counter[next_item] += 1
            if repeat_counter[next_item] > self.max_repeat:
                break

            path.append(next_item)

        if self.final_exam_id is not None and self.final_exam_id not in path:
            path.append(self.final_exam_id)

        return path

    def print_transition_counts(self):
        print(f"Transiciones observadas (conteo): {len(self.transition_counts)}")
