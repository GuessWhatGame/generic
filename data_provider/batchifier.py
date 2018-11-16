import copy
from enum import Enum


class BatchifierSplitMode(Enum):
    NoSplit = 0
    SingleQuestion = 1
    DialogueHistory = 2

    @staticmethod
    def from_string(s):
        s = s.lower()
        if s == "no_split":
            return BatchifierSplitMode.NoSplit
        elif s == "single_question":
            return BatchifierSplitMode.SingleQuestion
        elif s == "dialogue_history":
            return BatchifierSplitMode.SingleQuestion
        else:
            assert False, "Invalid question type for batchifier. Was {}".format(s)


def batchifier_split_helper(games, split_mode):

    new_games = []

    # One sample = One full dialogue
    if split_mode == BatchifierSplitMode.NoSplit:
        new_games = games

    # One sample = One question
    elif split_mode == BatchifierSplitMode.SingleQuestion:
        for game in games:
            for i, q, a in zip(game.question_ids, game.questions, game.answers):
                new_game = copy.copy(game)
                new_game.questions = [q]
                new_game.question_ids = [i]
                new_game.answers = [a]
                new_game.is_full_dialogue = False

                new_games.append(new_game)


    # One sample = Subset of questions
    elif split_mode == BatchifierSplitMode.DialogueHistory:
        for game in games:
            for i in range(len(game.question_ids)):
                new_game = copy.copy(game)
                new_game.questions = game.questions[:i + 1]
                new_game.question_ids = game.question_ids[:i + 1]
                new_game.answers = game.answers[:i + 1]
                new_game.is_full_dialogue = len(game.question_ids) == len(new_game.question_ids)

                new_games.append(new_game)

    return new_games


class AbstractBatchifier(object):

    def split(self, games):
        return games

    def filter(self, games):
        return games

    def apply(self, games):
        return games
