"""Module for exceptions in the CompoundMapper package"""



class BioactivitiesNotFoundError(Exception):
    """Error called when no bioactivities are found that satisfy the given query parameters"""

    def __init__(
        self,
        message="No bioactivities found that satisfy the given query parameters",
        parameters: dict | None = None,
    ):
        self.message = f"{message}"
        if parameters is not None:
            self.message += ": {" + ", ".join(f"{k}:{v}" for k, v in parameters.items()) + "}"
        super().__init__(self.message)

    pass
