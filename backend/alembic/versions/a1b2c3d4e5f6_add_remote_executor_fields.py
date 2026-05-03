"""add_remote_executor_fields

Revision ID: a1b2c3d4e5f6
Revises: 780489bdb99d
Create Date: 2026-05-03 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '780489bdb99d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('executor', sa.String(), nullable=False, server_default='local'))
        batch_op.add_column(sa.Column('remote_url_snapshot', sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_column('remote_url_snapshot')
        batch_op.drop_column('executor')
