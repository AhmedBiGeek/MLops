"""Enregistre les flows Prefect et les lance une fois par jour."""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all, flow_evaluate, flow_train

TIMEZONE = "Africa/Tunis"


def daily(hour: int, minute: int = 0) -> CronSchedule:
    """Construit un cron quotidien à l'heure de Tunis.

    Args:
        hour: Heure de lancement, de 0 à 23.
        minute: Minute de lancement, de 0 à 59.

    Returns:
        Planning Prefect.
    """
    return CronSchedule(cron=f"{minute} {hour} * * *", timezone=TIMEZONE)


if __name__ == "__main__":
    # all à 02:00, train à 02:30, evaluate à 03:00 pour éviter
    # que les trois flows écrivent les mêmes fichiers en même temps.
    serve(
        flow_all.to_deployment(
            name="ml-pipeline-all",
            schedules=[daily(2, 0)],
            tags=["mlops", "full-pipeline"],
        ),
        flow_train.to_deployment(
            name="ml-pipeline-train",
            schedules=[daily(2, 30)],
            tags=["mlops", "train"],
        ),
        flow_evaluate.to_deployment(
            name="ml-pipeline-evaluate",
            schedules=[daily(3, 0)],
            tags=["mlops", "evaluate"],
        ),
    )
