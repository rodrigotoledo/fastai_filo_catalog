"""Rename tables to plural: clients, client_addresses, photos

Revision ID: e41abfc426c3
Revises: 9021435bb5e9
Create Date: 2025-12-02 19:47:36.003200

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e41abfc426c3'
down_revision: Union[str, Sequence[str], None] = '9021435bb5e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Rename tables to plural
    op.rename_table('client', 'clients')
    op.rename_table('clientaddress', 'client_addresses')
    op.rename_table('photo', 'photos')

    # Rename indexes
    op.execute('ALTER INDEX ix_client_email RENAME TO ix_clients_email')
    op.execute('ALTER INDEX ix_client_cpf RENAME TO ix_clients_cpf')


def downgrade() -> None:
    """Downgrade schema."""
    # Rename indexes back
    op.execute('ALTER INDEX ix_clients_email RENAME TO ix_client_email')
    op.execute('ALTER INDEX ix_clients_cpf RENAME TO ix_client_cpf')

    # Rename tables back to singular
    op.rename_table('photos', 'photo')
    op.rename_table('client_addresses', 'clientaddress')
    op.rename_table('clients', 'client')
