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

    users = []
    for student in students:
        users.append(
            User(
                id=student["id"],
                username=student["username"],
                email=student["email"],
                password="admin",
            )
        )
    client.create_users(users)
    print(f"Added {len(users)} students")

    # Add items data
    with open("database/data/vcourse/resources.json") as f:
        items = json.load(f)

    resources = []
    for item in items:
        if item["type"] == "quiz":
            resources.append(
                Resource(
                    id=item["id"],
                    recid=item["recid"],
                    name=item["name"],
                    type=item["type"],
                    quizid=item["quizid"],
                )
            )
        else:
            resources.append(
                Resource(
                    id=item["id"],
                    recid=item["recid"],
                    name=item["name"],
                    type=item["type"],
                )
            )
    client.create_resources(resources)
    print(f"Added {len(resources)} items")

    # Add interactions data
    # with open('database/logs.json') as f:
    #     logs = json.load(f)

    interactions = []
    for student in students:
        interactions.append(
            Interaction(timestamp="1726814835", user_id=student["id"], resource_id=0)
        )
    client.insert_interactions(interactions)


if __name__ == "__main__":
    main()
