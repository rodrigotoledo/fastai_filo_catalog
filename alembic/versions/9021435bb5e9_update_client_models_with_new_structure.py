"""Update client models with new structure

Revision ID: 9021435bb5e9
Revises: 9f945150e775
Create Date: 2025-12-02 17:10:38.806884

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9021435bb5e9'
down_revision: Union[str, Sequence[str], None] = '9f945150e775'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Adicionar nickname e rg à tabela client
    op.add_column('client', sa.Column('nickname', sa.String(), nullable=True))
    op.add_column('client', sa.Column('rg', sa.String(), nullable=True))

    # Remover gender e notes da tabela client
    op.drop_column('client', 'gender')
    op.drop_column('client', 'notes')

    # Renomear address_type para type na tabela clientaddress
    op.alter_column('clientaddress', 'address_type', new_column_name='type')

    # Remover country e is_default da tabela clientaddress
    op.drop_column('clientaddress', 'country')
    op.drop_column('clientaddress', 'is_default')


def downgrade() -> None:
    """Downgrade schema."""
    # Reverter as mudanças
    op.add_column('clientaddress', sa.Column('is_default', sa.Boolean(), nullable=False, default=False))
    op.add_column('clientaddress', sa.Column('country', sa.String(), nullable=False, default='Brasil'))
    op.alter_column('clientaddress', 'type', new_column_name='address_type')

    op.add_column('client', sa.Column('notes', sa.String(), nullable=True))
    op.add_column('client', sa.Column('gender', sa.String(), nullable=True))

    op.drop_column('client', 'rg')
    op.drop_column('client', 'nickname')
