class CameraStatus :
    is_enable = True

    @classmethod
    def disable(cls):
        cls.is_enable = False

    @classmethod
    def enable(cls):
        cls.is_enable = True

    @classmethod
    def getstatus(cls):
        return cls.is_enable
