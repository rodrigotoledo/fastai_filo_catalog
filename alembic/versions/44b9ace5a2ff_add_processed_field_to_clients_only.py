"""add_processed_field_to_clients_only

Revision ID: 44b9ace5a2ff
Revises: e69eeb286029
Create Date: 2025-12-05 02:16:23.049566

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '44b9ace5a2ff'
down_revision: Union[str, Sequence[str], None] = 'e69eeb286029'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('clients', sa.Column('processed', sa.Boolean(), server_default=sa.text('false'), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('clients', 'processed')
