"""remove_ai_fields_from_clients

Revision ID: 2c1fcf220738
Revises: d9336f00dd07
Create Date: 2025-12-04 22:40:45.724786

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2c1fcf220738'
down_revision: Union[str, Sequence[str], None] = 'd9336f00dd07'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Remove AI-related columns from clients table
    op.drop_column('clients', 'processed')
    op.drop_column('clients', 'embedding')
    op.drop_column('clients', 'ai_description')
    op.drop_column('clients', 'user_description')


def downgrade() -> None:
    """Downgrade schema."""
    # Add back AI-related columns (for rollback)
    op.add_column('clients', sa.Column('processed', sa.Boolean(), nullable=False, default=False))
    op.add_column('clients', sa.Column('embedding', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('clients', sa.Column('ai_description', sa.Text(), nullable=True))
    op.add_column('clients', sa.Column('user_description', sa.Text(), nullable=True))
