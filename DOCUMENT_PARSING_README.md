# Document Parsing Feature

## Overview
The Photo Finder API now supports automatic client creation from various document formats using AI-powered text extraction.

## Supported File Formats
- **PDF** (.pdf) - Text extraction using PyPDF2 and pdfplumber
- **Word Documents** (.docx) - Text extraction using python-docx
- **Images** (.png, .jpg, .jpeg, .tiff, .bmp) - OCR using pytesseract
- **CSV Files** (.csv) - Data parsing using pandas
- **Excel Files** (.xlsx, .xls) - Data parsing using pandas/openpyxl
- **Markdown** (.md) - Text extraction with formatting cleanup
- **Plain Text** (.txt) - Direct text reading

## API Endpoint

### POST `/api/v1/clients/upload-document`

Upload and process a document to extract client information.

**Parameters:**
- `file` (required): The document file to process
- `create_client` (optional): Boolean flag to automatically create client if data is valid

**Response:**
```json
{
  "filename": "document.pdf",
  "file_size": 12345,
  "file_type": "PDF",
  "extracted_data": {
    "name": "João Silva",
    "cpf": "123.456.789-00",
    "email": "joao.silva@email.com",
    "phone": "(11) 99999-9999",
    "date_of_birth": null,
    "address": {
      "street": "Rua das Flores",
      "number": "123",
      "city": "São Paulo",
      "state": "SP",
      "postal_code": "01234-567"
    },
    "notes": "Additional extracted information...",
    "confidence": "high"
  },
  "validation_errors": [],
  "is_valid": true,
  "processing_status": "success",
  "client_created": true,
  "created_client": { /* Client object if created */ }
}
```

## Processing Flow

1. **File Validation**: Check file type and size (max 10MB)
2. **Text Extraction**: Extract text using appropriate library for file type
3. **AI Processing**: Use CLIP-based text analysis to identify client data patterns
4. **Data Validation**: Validate extracted CPF, email, phone formats
5. **Client Creation** (optional): Create client record with extracted data

## Data Extraction Patterns

The system recognizes common Brazilian document patterns:
- **Names**: "Nome: João Silva" or "Cliente: Maria Santos"
- **CPF**: "CPF: 123.456.789-00" or "123.456.789-00"
- **Email**: Standard email format recognition
- **Phone**: "(11) 99999-9999" or "11 99999-9999"
- **Addresses**: "Rua X, 123, Cidade - UF, CEP: 12345-678"

## Validation Rules

- **CPF**: Brazilian format validation (relaxed for test data)
- **Email**: Standard email format
- **Phone**: Brazilian phone number format (10-11 digits)
- **Required Fields**: Name is mandatory for client creation

## Usage Examples

### Extract Data Only
```bash
curl -X POST http://localhost:8000/api/v1/clients/upload-document \
  -F "file=@client_document.pdf" \
  -F "create_client=false"
```

### Extract and Create Client
```bash
curl -X POST http://localhost:8000/api/v1/clients/upload-document \
  -F "file=@client_document.pdf" \
  -F "create_client=true"
```

## Error Handling

- **Unsupported file type**: Returns 400 with supported formats list
- **File too large**: Returns 400 (max 10MB)
- **Text extraction failed**: Returns 500 with error details
- **Invalid extracted data**: Returns validation errors in response
- **Client creation failed**: Returns creation error details

## Dependencies

The following packages were added for document processing:
- `PyPDF2==3.0.1` - PDF text extraction
- `python-docx==1.2.0` - Word document processing
- `pytesseract==0.3.13` - OCR for images
- `pdfplumber==0.11.8` - Advanced PDF processing
- `pandas==2.3.3` - CSV/Excel data parsing
- `openpyxl==3.1.5` - Excel file support
- `Pillow` - Image processing (already included)

## Custom Extraction Prompts

You can now provide custom extraction prompts to guide the AI in extracting specific information from documents:

### Examples of Custom Prompts

```bash
# For resumes/CVs
"Este é um currículo profissional. Extraia o nome completo, email, telefone e localização da pessoa."

# For business documents
"Este documento contém informações de cliente empresarial. Extraia nome da empresa, CNPJ, email de contato e endereço comercial."

# For personal documents
"Documento pessoal brasileiro. Procure por nome, CPF, data de nascimento e endereço residencial."

# For invoices
"Esta é uma fatura/nota fiscal. Extraia nome do cliente, CNPJ/CPF, valor total e data de emissão."
```

### How It Works

1. **Text Extraction**: Document content is extracted using appropriate libraries
2. **AI-Guided Analysis**: Your custom prompt instructs the AI what to look for
3. **Structured Output**: Information is extracted into standardized client fields
4. **Validation**: Extracted data is validated before client creation

### Prompt Guidelines

- **Be specific**: Tell the AI what type of document it is
- **List fields**: Specify exactly what information you want extracted
- **Use context**: Provide context about the document's purpose or origin
- **Language**: Use Portuguese for Brazilian documents, English for international ones
