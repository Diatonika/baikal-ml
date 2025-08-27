from polars import Expr, Int32, duration, lit


def duration_components(column: str | Expr) -> dict[str, Expr]:
    expression = lit(column) if isinstance(column, str) else column

    return {
        "w": expression.str.extract(r"(\d+?)w\b").cast(Int32).fill_null(0),
        "d": expression.str.extract(r"(\d+?)d\b").cast(Int32).fill_null(0),
        "h": expression.str.extract(r"(\d+?)h\b").cast(Int32).fill_null(0),
        "m": expression.str.extract(r"(\d+?)m\b").cast(Int32).fill_null(0),
        "s": expression.str.extract(r"(\d+?)s\b").cast(Int32).fill_null(0),
        "ms": expression.str.extract(r"(\d+?)ms\b").cast(Int32).fill_null(0),
        "us": expression.str.extract(r"(\d+?)us\b").cast(Int32).fill_null(0),
        "ns": expression.str.extract(r"(\d+?)ns\b").cast(Int32).fill_null(0),
    }


def as_duration(column: str | Expr) -> Expr:
    components = duration_components(column)

    return duration(
        weeks=components["w"],
        days=components["d"],
        hours=components["h"],
        minutes=components["m"],
        seconds=components["s"],
        milliseconds=components["ms"],
        microseconds=components["us"],
        nanoseconds=components["ns"],
    )
