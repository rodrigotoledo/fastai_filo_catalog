"""add_user_description_to_photos

Revision ID: 89bc2dc5b677
Revises: 3f7142bce537
Create Date: 2025-12-01 00:25:45.462432

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '89bc2dc5b677'
down_revision: Union[str, Sequence[str], None] = '3f7142bce537'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Adicionar coluna para descrição fornecida pelo usuário
    op.add_column('photo', sa.Column('user_description', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remover coluna
    op.drop_column('photo', 'user_description')
