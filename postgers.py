import psycopg2

# Database connection details
db_config = {
    "host": "kms-postgres.postgres.database.azure.com",
    "dbname": "feedback",
    "user": "kmsadmin",
    "password": "Celebal@123456",
    "port": 5432,  # Default PostgreSQL port
    "sslmode": "require",  # Enforce SSL
}

# SQL to create the feedback table
create_table_query = """
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    feedback_text TEXT NOT NULL,
    rating INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

try:
    # Connect to PostgreSQL
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:
            # Create feedback table
            cur.execute(create_table_query)
            print("Feedback table created successfully.")
except psycopg2.Error as e:
    print(f"An error occurred: {e}")


# Feedback to store
feedback_data = {
    "user_id": 1,
    "feedback_text": "This is a great service!",
    "rating": 5
}

# SQL to insert feedback
insert_query = """
INSERT INTO feedback (user_id, feedback_text, rating)
VALUES (%s, %s, %s);
"""

try:
    # Connect to PostgreSQL
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:
            # Insert feedback data
            cur.execute(insert_query, (feedback_data["user_id"], feedback_data["feedback_text"], feedback_data["rating"]))
            conn.commit()  # Commit transaction
            print("Feedback inserted successfully.")
except psycopg2.Error as e:
    print(f"An error occurred: {e}")

# SQL to query feedback
select_query = "SELECT * FROM feedback;"

try:
    # Connect to PostgreSQL
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:
            # Fetch all feedback records
            cur.execute(select_query)
            rows = cur.fetchall()
            for row in rows:
                print(row)
except psycopg2.Error as e:
    print(f"An error occurred: {e}")
