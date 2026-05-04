"""subtitle_unique_constraint_and_workdir_token

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Deduplicate subtitle rows: keep the most-recently-created row per (job_id, format, language)
    # Uses a join-based approach that works on all SQLite versions
    op.execute("""
        DELETE FROM subtitles
        WHERE id NOT IN (
            SELECT s.id FROM subtitles s
            INNER JOIN (
                SELECT job_id, format, COALESCE(language, '') AS lang_key, MAX(created_at) AS max_date
                FROM subtitles
                GROUP BY job_id, format, COALESCE(language, '')
            ) latest
            ON s.job_id = latest.job_id
               AND s.format = latest.format
               AND COALESCE(s.language, '') = latest.lang_key
               AND s.created_at = latest.max_date
        )
    """)

    # Add unique index on (job_id, format, language)
    op.create_index(
        "uq_subtitle_job_fmt_lang",
        "subtitles",
        ["job_id", "format", "language"],
        unique=True,
    )

    # Add workdir_token to jobs (for two-stage remote handoff)
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('workdir_token', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_index("uq_subtitle_job_fmt_lang", "subtitles")
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_column('workdir_token')
