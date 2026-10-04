import json
import os


# ==========================================
# METADATA STORAGE
# ==========================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
DATA_FOLDER = os.getenv(
    "DATA_FOLDER",
    os.path.join(PROJECT_ROOT, "data")
)

METADATA_FILE = os.path.join(
    DATA_FOLDER,
    "document_metadata.json"
)


# ==========================================
# DEFAULT DOCUMENT METADATA
# ==========================================

DEFAULT_METADATA = {

    "PYTHON.pdf": {
        "subject": "Python",
        "chapter": "General",
        "content_type": "Notes"
    },

    "c_programming_basics.pdf": {
        "subject": "C-Programming",
        "chapter": "General",
        "content_type": "Notes"
    },

    "Python Question Bank.pdf": {
        "subject": "Python",
        "chapter": "General",
        "content_type": "Question Bank"
    },

    "2919.pdf": {
        "subject": "General",
        "chapter": "General",
        "content_type": "Notes"
    },

    "DM UNIT3.pptx": {
        "subject": "DM",
        "chapter": "UNIT 3",
        "content_type": "Notes"
    }
}


# ==========================================
# LOAD METADATA
# ==========================================

def load_metadata():

    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    if not os.path.exists(
        METADATA_FILE
    ):

        save_metadata(
            DEFAULT_METADATA
        )

        return DEFAULT_METADATA.copy()


    try:

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)


        # Make sure default documents
        # still exist in metadata
        changed = False

        for filename, info in DEFAULT_METADATA.items():

            if filename not in metadata:

                metadata[filename] = info
                changed = True


        if changed:

            save_metadata(
                metadata
            )


        return metadata


    except (
        json.JSONDecodeError,
        OSError
    ):

        print(
            "Warning: Could not read document metadata."
        )

        print(
            "Recreating metadata file..."
        )

        save_metadata(
            DEFAULT_METADATA
        )

        return DEFAULT_METADATA.copy()


# ==========================================
# SAVE METADATA
# ==========================================

def save_metadata(metadata):

    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================
# ADD / UPDATE DOCUMENT METADATA
# ==========================================

def set_document_metadata(
    filename,
    subject,
    chapter,
    content_type
):

    metadata = load_metadata()

    metadata[filename] = {

        "subject":
            subject,

        "chapter":
            chapter,

        "content_type":
            content_type
    }

    save_metadata(
        metadata
    )

    return metadata[filename]


# ==========================================
# GET ONE DOCUMENT
# ==========================================

def get_document_metadata(
    filename
):

    metadata = load_metadata()

    return metadata.get(
        filename,
        {
            "subject": "General",
            "chapter": "General",
            "content_type": "Notes"
        }
    )


# ==========================================
# GET ALL DOCUMENTS
# ==========================================

def get_all_metadata():

    return load_metadata()