import os
import json
from datetime import datetime
from dotenv import load_dotenv

import sys

sys.path.append(".")
from database.orm.query_generator import QueryGenerator
from database.models.db import Interaction, User, Resource

load_dotenv(override=True)

client = QueryGenerator()


def main() -> None:
    client.connect()

    # Add users data
    with open("database/data/students.json") as f:
        students = json.load(f)

    for student in students:
        client.create_user(
            User(
                id=student["id"],
                username=student["username"],
                email=student["email"],
                password="admin",
            )
        )

    client.create_user(
        User(id=0, username="admin", email="admin", password="admin")
    )

    # Add items data
    with open("database/data/fakecourse/resources.json") as f:
        items = json.load(f)

    for item in items:
        if item["type"] == "quiz":
            client.create_resource(
                Resource(
                    id=item["id"],
                    recid=item["recid"],
                    name=item["name"],
                    type=item["type"],
                    quizid=item["quizid"],
                )
            )
        else:
            client.create_resource(
                Resource(
                    id=item["id"],
                    recid=item["recid"],
                    name=item["name"],
                    type=item["type"],
                )
            )

    # Add interactions data
    # with open('database/logs.json') as f:
    #     logs = json.load(f)

    for student in students:
        client.insert_interaction(
            Interaction(timestamp="1726814835", user_id=student["id"], resource_id=0)
        )

    # for log in logs:
    #     client.insert_interaction(Interaction(timestamp=log['timestamp'], user_id=log['user_id'], resource_id=log['resource_id']))

    # os.remove('database/students.json')
    # os.remove('database/resources.json')
    # os.remove('database/logs.json')


if __name__ == "__main__":
    main()
