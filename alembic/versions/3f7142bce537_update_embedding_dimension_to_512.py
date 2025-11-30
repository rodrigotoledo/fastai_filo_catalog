"""update_embedding_dimension_to_512

Revision ID: 3f7142bce537
Revises: bb584c27fa44
Create Date: 2025-11-30 19:43:16.927371

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3f7142bce537'
down_revision: Union[str, Sequence[str], None] = 'bb584c27fa44'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Primeiro, limpar embeddings existentes (vamos reprocessar tudo)
    op.execute("UPDATE photo SET embedding = NULL WHERE embedding IS NOT NULL")

    # Alterar dimensão do vector de 768 para 512
    op.execute("ALTER TABLE photo ALTER COLUMN embedding TYPE vector(512)")


def downgrade() -> None:
    """Downgrade schema."""
    # Primeiro, limpar embeddings existentes
    op.execute("UPDATE photo SET embedding = NULL WHERE embedding IS NOT NULL")

    # Reverter para 768
    op.execute("ALTER TABLE photo ALTER COLUMN embedding TYPE vector(768)")
