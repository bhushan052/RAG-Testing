import os
import tempfile

from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from models.schemas import (
    SearchRequest,
    AskRequest
)

from services.document_loader import (
    extract_text
)

from services.chunking import (
    chunk_text
)

from services.embeddings import (
    create_embedding
)

from services.qdrant import (
    create_collection,
    insert_vector,
    search_vector
)

from services.llm import (
    generate_answer
)


app = FastAPI(
    title="Document RAG API",
    description="PDF and DOCX based Retrieval Augmented Generation API",
    version="1.0.0"
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Document RAG API is running"
    }


# --------------------------------------------------
# UPLOAD DOCUMENT
# --------------------------------------------------

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".pdf",
        ".docx"
    }

    extension = Path(
        file.filename
    ).suffix.lower()


    if extension not in allowed_extensions:

        raise HTTPException(

            status_code=400,

            detail=(
                "Unsupported file type. "
                "Only PDF and DOCX files are supported."
            )
        )


    contents = await file.read()


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp_file:

        temp_file.write(contents)

        temp_file_path = temp_file.name


    try:

        # ------------------------------------------
        # Step 1: Extract text
        # ------------------------------------------

        text = extract_text(
            temp_file_path
        )


        if not text.strip():

            raise HTTPException(

                status_code=400,

                detail=(
                    "No readable text was found "
                    "in the uploaded document."
                )
            )


        # ------------------------------------------
        # Step 2: Chunk document
        # ------------------------------------------

        chunks = chunk_text(
            text
        )


        # ------------------------------------------
        # Step 3: Create embeddings
        # ------------------------------------------

        chunks_created = 0


        for chunk in chunks:

            vector = create_embedding(
                chunk
            )


            # --------------------------------------
            # Step 4: Create Qdrant collection
            # --------------------------------------

            create_collection(
                vector_size=len(vector)
            )


            # --------------------------------------
            # Step 5: Store vector + metadata
            # --------------------------------------

            insert_vector(

                vector=vector,

                text=chunk,

                filename=file.filename
            )


            chunks_created += 1


        return {

            "message": (
                "Document uploaded and indexed successfully"
            ),

            "filename": file.filename,

            "chunks_created": chunks_created
        }


    finally:

        os.remove(
            temp_file_path
        )


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

@app.post("/search")
def search(
    request: SearchRequest
):

    query_vector = create_embedding(
        request.query
    )


    results = search_vector(

        query_vector=query_vector,

        top_k=request.top_k
    )


    return {

        "query": request.query,

        "results": [

            {
                "score": result.score,

                "text": result.payload.get(
                    "text",
                    ""
                ),

                "filename": result.payload.get(
                    "filename",
                    ""
                )
            }

            for result in results
        ]
    }


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.post("/ask")
def ask(
    request: AskRequest
):

    # ----------------------------------------------
    # Step 1: Create query embedding
    # ----------------------------------------------

    query_vector = create_embedding(
        request.query
    )


    # ----------------------------------------------
    # Step 2: Retrieve relevant chunks
    # ----------------------------------------------

    results = search_vector(

        query_vector=query_vector,

        top_k=request.top_k
    )


    # ----------------------------------------------
    # Step 3: Build context
    # ----------------------------------------------

    context_parts = []


    for result in results:

        chunk_text_value = result.payload.get(
            "text",
            ""
        )

        filename = result.payload.get(
            "filename",
            ""
        )


        context_parts.append(

            f"""
Source: {filename}

Content:
{chunk_text_value}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # ----------------------------------------------
    # Step 4: Generate answer
    # ----------------------------------------------

    answer = generate_answer(

        context=context,

        query=request.query
    )


    # ----------------------------------------------
    # Step 5: Return response
    # ----------------------------------------------

    return {

        "query": request.query,

        "answer": answer,

        "sources": [

            {
                "filename": result.payload.get(
                    "filename",
                    ""
                ),

                "score": result.score
            }

            for result in results
        ]
    }