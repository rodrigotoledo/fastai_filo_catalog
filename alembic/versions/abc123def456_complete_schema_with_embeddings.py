"""complete_schema_with_embeddings

Revision ID: abc123def456
Revises:
Create Date: 2025-12-08 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'abc123def456'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Criar extensão pgvector se não existir
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Criar tabela clients
    op.create_table('clients',
        sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('nickname', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('email', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('phone', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('cpf', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('rg', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('birth_date', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
        sa.Column('is_active', sa.BOOLEAN(), server_default=sa.text('true'), autoincrement=False, nullable=True),
        sa.Column('processed', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=True),
        sa.Column('embedding', postgresql.VECTOR(dim=512), autoincrement=False, nullable=True),
        sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=True),
        sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint('id', name='clients_pkey'),
        sa.UniqueConstraint('cpf', name='clients_cpf_key'),
        sa.UniqueConstraint('email', name='clients_email_key')
    )

    # Criar tabela client_addresses
    op.create_table('client_addresses',
        sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column('client_id', sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column('type', sa.VARCHAR(), server_default=sa.text("'Pessoal'::character varying"), autoincrement=False, nullable=True),
        sa.Column('street', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('number', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('complement', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('neighborhood', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('city', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('state', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('zip_code', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=True),
        sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=True),
        sa.ForeignKeyConstraint(['client_id'], ['clients.id'], name='client_addresses_client_id_fkey', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name='client_addresses_pkey')
    )

    # Garantir que a tabela photos tenha todos os campos necessários
    op.add_column('photos', sa.Column('embedding', postgresql.VECTOR(dim=512), nullable=True))
    op.add_column('photos', sa.Column('image_embedding', postgresql.VECTOR(dim=512), nullable=True))
    op.add_column('photos', sa.Column('description', sa.TEXT(), nullable=True))
    op.add_column('photos', sa.Column('user_description', sa.TEXT(), nullable=True))
    op.add_column('photos', sa.Column('gemini_file_id', sa.VARCHAR(), nullable=True))
    op.add_column('photos', sa.Column('image_data', postgresql.BYTEA(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remover colunas adicionadas à tabela photos
    op.drop_column('photos', 'image_data')
    op.drop_column('photos', 'gemini_file_id')
    op.drop_column('photos', 'user_description')
    op.drop_column('photos', 'description')
    op.drop_column('photos', 'image_embedding')
    op.drop_column('photos', 'embedding')

    # Dropar tabelas
    op.drop_table('client_addresses')
    op.drop_table('clients')

    # Remover extensão (opcional)
    op.execute("DROP EXTENSION IF EXISTS vector")
