class Mood:
    name = ""

    @classmethod
    def update_name(cls, mood_result):
        cls.name = mood_result

    @classmethod
    def get_name(cls):
        return cls.name
