import os
import json
import yaml
import smtplib
import ssl
import hashlib
import shutil
from email.message import EmailMessage
import streamlit as st
import base64
import io
import tempfile
import time
from pydantic import BaseModel

from pdf2image import convert_from_bytes  # pip install pdf2image
from openai import OpenAI

# ----------------------------
# 1) Load Config & Secrets
# ----------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

INVOICE_ANALYSIS_PROMPT = config_data["invoice_analysis_prompt"]
CONTACTS = config_data["contacts"]  # single list of contacts
GRANTS = config_data["grants"]  # single list of grants

with open("secret.yaml", "r", encoding="utf-8") as f:
    secrets_data = yaml.safe_load(f)

OPENAI_APIKEY = secrets_data["openai_api_key"]
SENDER_USERNAME = secrets_data["sender_username"]  # e.g. "myaccount@gmail.com"
SENDER_PASSWORD = secrets_data["sender_password"]  # MUST be an App Password


# ----------------------------
# 2) GPT-4o Invoice Parsing
# ----------------------------
class Invoice(BaseModel):
    amount: float
    rationale: str


# Instantiate the special "gpt-4o" client
client = OpenAI(api_key=OPENAI_APIKEY)


def encode_image(image_path):
    """
    Utility to read an image file from disk and return its base64-encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_invoice_pdf_with_gpt4o(uploaded_pdf):
    """
    1) Convert PDF to images (one per page).
    2) For each page, call gpt-4o-mini with the image.
    3) Combine the JSON results to find a final amount and rationale.
    """
    pdf_bytes = uploaded_pdf.read()  # read the entire PDF
    uploaded_pdf.seek(0)  # reset pointer so we can attach it later

    # Convert PDF pages to PIL images
    images = convert_from_bytes(pdf_bytes)

    # Create a temporary directory that will be cleaned up automatically
    with tempfile.TemporaryDirectory() as tmp_dir:
        # We'll collect amounts/rationales from each page
        amounts = []
        rationales = []

        for idx, img in enumerate(images):
            # Save the page as an image in the temporary directory
            page_path = os.path.join(tmp_dir, f"page_{idx+1}.png")
            img.save(page_path, "PNG")
            print(f"Saved page {idx+1} to {page_path}")

            # Encode the image in Base64
            base64_image = encode_image(page_path)

            # Make the request to the "gpt-4o-mini" model
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": INVOICE_ANALYSIS_PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                response_format=Invoice,
            )

            # Parse out the GPT result
            page_result = response.choices[0].message.content.strip()
            print(f"Page {idx+1} result: {page_result}")

            try:
                # Expect valid JSON with "amount" and "rationale"
                parsed = json.loads(page_result)
                amt_str = parsed.get("amount", "0.0")
                rat = parsed.get("rationale", "")
                amounts.append(float(amt_str))
                rationales.append(rat)
            except Exception as e:
                amounts.append(0.0)
                rationales.append(f"Error parsing page {idx+1}: {e}")

        # Combine or pick the "largest" amount found across pages (naive approach)
        final_amount = max(amounts) if amounts else 0.0
        # Join rationales for all pages, or just pick the last
        final_rationale = " | ".join(rationales) if rationales else ""

    return final_amount, final_rationale


def parse_invoice_with_llm(uploaded_file):
    """
    Dispatch to either PDF parsing or single-image parsing with GPT-4o.
    """
    file_type = uploaded_file.type.lower()
    if file_type == "application/pdf":
        return parse_invoice_pdf_with_gpt4o(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF.")


# ---------------------------
# 3) Email Sending Function
# ---------------------------
def send_email_via_gmail(to_emails, cc_emails, subject, body, attachment_file=None):
    """
    Sends an email via Gmail with an optional attachment (PDF or image).

    Args:
        to_emails (list[str]): List of recipient emails (the "To" field).
        cc_emails (list[str]): List of CC emails.
        subject (str): Email subject.
        body (str): Email body.
        attachment_file: A file-like object to attach, or None.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_USERNAME  # Always from your Gmail account
    msg["To"] = ", ".join(to_emails)

    # Remove duplicates in CC while preserving order
    if cc_emails:
        unique_cc = []
        for c in cc_emails:
            if c and c not in unique_cc:
                unique_cc.append(c)
        if unique_cc:
            msg["Cc"] = ", ".join(unique_cc)

    # print(msg["To"])

    msg.set_content(body)

    # Attach the original file if provided
    if attachment_file:
        file_bytes = attachment_file.read()
        attachment_file.seek(0)
        filename = attachment_file.name or "invoice.pdf"
        msg.add_attachment(
            file_bytes,
            maintype="application",
            subtype="octet-stream",
            filename=filename,
        )
    # print("attachment added")
    #
    # F**k google, it needs 2FA
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_USERNAME, SENDER_PASSWORD)
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        # Often indicates a bad username/password or missing App Password setup
        raise Exception(f"SMTP Error: {e}")


# ---------------------------
# 4) Streamlit App
# ---------------------------
def run_streamlit_app():
    st.title("Simonlab PCard Receipt")

    contact_names = [c["name"] for c in CONTACTS]
    purchaser_names = [c["name"] for c in CONTACTS if c["role"] == "purchaser"]
    admin_names = [c["name"] for c in CONTACTS if c["role"] == "admin"]

    # ----------------------
    # Initialize Session Vars
    # ----------------------
    if "invoice_amount" not in st.session_state:
        st.session_state["invoice_amount"] = 0.0
    if "invoice_rationale" not in st.session_state:
        st.session_state["invoice_rationale"] = ""
    if "parsed_file_hash" not in st.session_state:
        st.session_state["parsed_file_hash"] = None

    # ----------------------
    # A) Select Purchaser
    # ----------------------
    # The purchaser is always one of the "purchaser_names".
    purchaser_name = st.selectbox("Purchaser", purchaser_names)
    purchaser_email = next(
        (x["email"] for x in CONTACTS if x["name"] == purchaser_name), ""
    )

    # ----------------------
    # B) Multi-select "To" recipients, ensuring purchaser is included
    # ----------------------
    # Let the user pick multiple recipients from contact_names
    selected_to_names = st.multiselect(
        "To Recipients",
        options=[name for name in admin_names],
        # default=[purchaser_name],
        help="Choose all recipients for the To field.",
    )
    # Force the purchaser to be in the list if not manually selected
    to_name_string = ", ".join(selected_to_names)
    selected_to_names.append(purchaser_name)

    # Convert these "To" names to emails
    to_emails = []
    for name in selected_to_names:
        c = next((x for x in CONTACTS if x["name"] == name), None)
        if c and c.get("email"):
            to_emails.append(c["email"])

    # ----------------------
    # C) Upload Invoice (PDF)
    # ----------------------
    uploaded_invoice = st.file_uploader("Upload Invoice (PDF)", type=["pdf"])

    # Parse with LLM if needed
    if uploaded_invoice is not None:
        file_bytes = uploaded_invoice.getvalue()
        new_file_hash = hashlib.md5(file_bytes).hexdigest()

        if st.session_state["parsed_file_hash"] != new_file_hash:
            # Parse again
            amount, rationale = parse_invoice_with_llm(uploaded_invoice)
            st.session_state["invoice_amount"] = amount
            st.session_state["invoice_rationale"] = rationale
            st.session_state["parsed_file_hash"] = new_file_hash
        else:
            # Reuse existing parse
            amount = st.session_state["invoice_amount"]
            rationale = st.session_state["invoice_rationale"]

        # Show results
        if st.session_state["invoice_amount"] > 0:
            st.success(
                f"**Extracted Amount:** ${st.session_state['invoice_amount']:,.2f}"
            )
        else:
            st.warning("No valid amount parsed.")

        if st.session_state["invoice_rationale"]:
            st.success(
                f"**Extracted Rationale:** {st.session_state['invoice_rationale']}"
            )
        else:
            st.warning("No rationale parsed.")

    # ----------------------
    # D) Grant Selection
    grant_names = [g["name"] for g in GRANTS]
    grant_selected_name = st.selectbox("Grant", grant_names)
    grant_code = next(
        (str(x["code"]) for x in GRANTS if x["name"] == grant_selected_name), ""
    )
    st.session_state["grant_code"] = grant_code

    # ----------------------
    # E) CC Recipients
    # ----------------------
    cc_selected_names = st.multiselect(
        "CC Recipients",
        options=[n for n in contact_names],
        default=["rydberglabreceipts"] if "rydberglabreceipts" in contact_names else [],
        help="Optional: pick additional recipients to CC.",
    )
    cc_emails = []
    for name in cc_selected_names:
        c = next((x for x in CONTACTS if x["name"] == name), None)
        if c and c.get("email"):
            cc_emails.append(c["email"])

    # ----------------------
    # F) Subject & Body
    # ----------------------
    subject = f"PCard_Simon Group_AMOUNT_{st.session_state['invoice_amount']:.2f}"
    subject_input = st.text_input("Email Subject", value=subject)

    default_body = (
        f"Hi {to_name_string}.\n\n"
        f"This is {purchaser_name} from Jon Simon group, we have a new purchase with PCard.\n\n"
        f"Amount: ${st.session_state['invoice_amount']:.2f}\n"
        f"Rationale: {st.session_state['invoice_rationale']}\n"
        f"Account: {grant_code}\n\n"
        f"Best,\n{purchaser_name}\n"
        f"---------------------\n"
        f"This is an automated email generated by LLM, please reply to {purchaser_name} ({purchaser_email}) for any questions.\n"
    )
    body_input = st.text_area("Email Body", default_body, height=240)

    # ----------------------
    # F) Send Email Button
    # ----------------------
    if st.button("Send Email"):
        if not uploaded_invoice:
            st.error("Please upload an invoice first.")
        else:
            if not to_emails:
                st.error("No valid To recipients. Please select at least one contact.")
            else:
                try:
                    st.info("Sending email...")
                    send_email_via_gmail(
                        to_emails=to_emails,
                        cc_emails=cc_emails,
                        subject=subject_input,
                        body=body_input,
                        attachment_file=uploaded_invoice,
                    )
                    st.success("Email sent successfully!")
                except Exception as e:
                    st.error(f"Error sending email: {e}")


# -----------
# Run App
# -----------
if __name__ == "__main__":
    run_streamlit_app()
