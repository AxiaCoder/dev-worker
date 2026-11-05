"""Wording of Pydantic validation errors as sentences, copied from the Broker worker."""

from pydantic import ValidationError


def describe_validation_error(error: ValidationError) -> list[str]:
    """Turn a Pydantic validation error into one single-line sentence per problem."""
    sentences: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"])
        message = detail["msg"].removesuffix(".")
        sentence = f"{location}: {message}." if location else f"{message}."
        sentences.append(" ".join(sentence.split()))
    return sentences
