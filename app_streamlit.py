import os
import json
import yaml
import smtplib
import ssl
import hashlib
from email.message import EmailMessage
import streamlit as st
import base64
import tempfile
from pydantic import BaseModel

from pdf2image import convert_from_bytes  # pip install pdf2image
from openai import OpenAI

# ----------------------------
# 1) Load Config & Secrets
# ----------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

INVOICE_ANALYSIS_PROMPT = config_data["invoice_analysis_prompt"]
SUMMARY_PROMPT = config_data["summary_prompt"]
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
    Read an image file from disk and return its base64-encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_invoice_pdf_with_gpt4o(uploaded_pdf):
    """
    1) Convert PDF to images (one per page).
    2) For each page, call GPT with the base64 image for invoice info.
    3) Sum up amounts & combine rationales for the entire PDF.
    """
    st.info(f"Processing invoices: {uploaded_pdf.name}")
    pdf_bytes = uploaded_pdf.read()
    uploaded_pdf.seek(0)  # reset pointer so we can re-attach the file

    images = convert_from_bytes(pdf_bytes)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # only process the first page
        # for idx, img in enumerate(images):
        # Save page to disk
        idx = 0
        img = images[0]
        page_path = os.path.join(tmp_dir, f"page_{idx+1}.png")
        img.save(page_path, "PNG")

        # Convert to base64
        base64_image = encode_image(page_path)

        # Send request to GPT-4o
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
        page_result = response.choices[0].message.content.strip()
    #
    try:
        parsed = json.loads(page_result)
        amt_str = parsed.get("amount", 0.0)
        rat = parsed.get("rationale", "")
        amount = float(amt_str)
        rationale = str(rat)
        st.success(f"**Parsed Amount:** ${amount:,.2f}")
        st.success(f"**Rationale:** {rationale}")

        return amount, rationale
    except Exception as e:
        st.error(f"Error parsing invoice: {e}")
        return 0.0, ""


def parse_invoice_with_llm(uploaded_file):
    """
    Dispatch to PDF parsing. Extend for other file types if needed.
    """
    if uploaded_file.type.lower() == "application/pdf":
        return parse_invoice_pdf_with_gpt4o(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF.")


def summarize_rationales_with_llm(rationales):
    """
    GPT call to produce a single-sentence summary describing the main purchase,
    including the manufacturer or brand, from multiple rationales.
    """
    st.info("Summarizing multiple invoices...")
    try:
        prompt_text = SUMMARY_PROMPT + "\n".join(f" - {rat}" for rat in rationales)

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
        )

        summary = response.choices[0].message.content.strip()
        # st.success(f"**LLM Summary:** {summary}")
    except Exception as e:
        st.error(f"Error summarizing rationales: {e}")
    return summary


def parse_multiple_invoices_with_llm(uploaded_files):
    """
    For multiple PDFs:
      - Parse each
      - Sum total amounts
      - If more than one PDF, produce a single-sentence summary of them all.
        Otherwise, just use that single PDF's rationale.
    """
    # Build hash to detect changes in the set of files
    hash_md5 = hashlib.md5()
    for f in uploaded_files:
        file_bytes = f.read()
        hash_md5.update(file_bytes)
        f.seek(0)
    new_files_hash = hash_md5.hexdigest()

    # Check if we already parsed these files
    if st.session_state.get("parsed_files_hash") == new_files_hash:
        # Return existing results
        return (
            st.session_state["invoice_amounts"],
            st.session_state["invoice_rationale"],
            new_files_hash,
        )

    # Otherwise, parse from scratch
    amounts = []
    rationales = []
    for f in uploaded_files:
        amount, rationale = parse_invoice_with_llm(f)
        amounts.append(amount)
        rationales.append(rationale)

    # If only one file, just use its single rationale
    if len(uploaded_files) == 1:
        return amounts, rationales[0], new_files_hash
    else:
        rationale_summary = summarize_rationales_with_llm(rationales)
        return amounts, rationale_summary, new_files_hash


# ---------------------------
# 3) Email Sending Function
# ---------------------------
def send_email_via_gmail(to_emails, cc_emails, subject, body, attachment_files=None):
    """
    Sends an email via Gmail with an optional list of attachments (PDFs, images, etc.).
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_USERNAME
    msg["To"] = ", ".join(to_emails)

    if cc_emails:
        unique_cc = []
        for c in cc_emails:
            if c and c not in unique_cc:
                unique_cc.append(c)
        if unique_cc:
            msg["Cc"] = ", ".join(unique_cc)

    msg.set_content(body)

    # Attach files
    if attachment_files:
        for f in attachment_files:
            file_bytes = f.read()
            f.seek(0)
            filename = f.name or "invoice.pdf"
            msg.add_attachment(
                file_bytes,
                maintype="application",
                subtype="octet-stream",
                filename=filename,
            )

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_USERNAME, SENDER_PASSWORD)
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
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
    if "invoice_amounts" not in st.session_state:
        st.session_state["invoice_amounts"] = []
    if "invoice_amount_sum" not in st.session_state:
        st.session_state["invoice_amount_sum"] = 0.0
    if "invoice_rationale" not in st.session_state:
        st.session_state["invoice_rationale"] = ""
    if "parsed_files_hash" not in st.session_state:
        st.session_state["parsed_files_hash"] = None

    # ----------------------
    # A) Select Purchaser
    # ----------------------
    purchaser_name = st.selectbox("Purchaser", purchaser_names)
    purchaser_email = next(
        (x["email"] for x in CONTACTS if x["name"] == purchaser_name), ""
    )

    # ----------------------
    # B) Multi-select "To" recipients (admin + purchaser)
    # ----------------------
    selected_to_names = st.multiselect(
        "To Recipients",
        options=admin_names,
        help="Choose all recipients for the To field.",
    )
    if purchaser_name not in selected_to_names:
        selected_to_names.append(purchaser_name)

    to_emails = []
    for name in selected_to_names:
        contact = next((x for x in CONTACTS if x["name"] == name), None)
        if contact and contact.get("email"):
            to_emails.append(contact["email"])

    # ----------------------
    # C) Upload multiple Invoices (PDFs)
    # ----------------------
    uploaded_invoices = st.file_uploader(
        "Upload Invoice(s) (PDF)", type=["pdf"], accept_multiple_files=True
    )

    # ----------------------
    # D) LLM Processing Button
    # Only parse after user clicks "Process with LLM"
    # ----------------------
    if st.button("Process with LLM"):
        if not uploaded_invoices:
            st.error("Please upload at least one invoice first.")
        else:
            # Parse + Summarize
            amounts, rationale, new_files_hash = parse_multiple_invoices_with_llm(
                uploaded_invoices
            )

            # Store results
            st.session_state["invoice_amounts"] = amounts
            st.session_state["invoice_amount_sum"] = sum(amounts)
            st.session_state["invoice_rationale"] = rationale
            st.session_state["parsed_files_hash"] = new_files_hash

            # Show results
            if len(amounts) > 1:
                st.success(
                    f"**Combined Amount:** ${st.session_state['invoice_amount_sum']:,.2f}"
                )
                st.success(f"**Rationale**: {rationale}")

    # ----------------------
    # E) Grant Selection
    # ----------------------
    grant_names = [g["name"] for g in GRANTS]
    grant_selected_name = st.selectbox("Grant", grant_names)
    grant_code = next(
        (str(x["code"]) for x in GRANTS if x["name"] == grant_selected_name), ""
    )
    st.session_state["grant_code"] = grant_code

    # ----------------------
    # F) CC Recipients
    # ----------------------
    cc_selected_names = st.multiselect(
        "CC Recipients",
        options=contact_names,
        default=["rydberglabreceipts"] if "rydberglabreceipts" in contact_names else [],
        help="Optional: pick additional recipients to CC.",
    )
    cc_emails = []
    for name in cc_selected_names:
        c = next((x for x in CONTACTS if x["name"] == name), None)
        if c and c.get("email"):
            cc_emails.append(c["email"])

    # ----------------------
    # G) Invoice Amount
    # ----------------------
    # display <text_input> "+" <text_input> for user to update
    len_invoice_amounts = len(st.session_state["invoice_amounts"])
    if len_invoice_amounts > 0:
        text_input_row = st.columns(
            len_invoice_amounts * 2 + 1, gap="small", vertical_alignment="center"
        )
        text_input_vals = []
        for idx in range(len_invoice_amounts):
            uploaded_fileName = uploaded_invoices[idx].name
            val = text_input_row[2 * idx].text_input(
                uploaded_fileName,
                value=f"{st.session_state['invoice_amounts'][idx]:.2f}",
            )
            text_input_vals.append(val)
            if idx < len_invoice_amounts - 1:
                text_input_row[2 * idx + 1].text("+")
        st.session_state["invoice_amounts"] = [float(val) for val in text_input_vals]
        st.session_state["invoice_amount_sum"] = sum(
            st.session_state["invoice_amounts"]
        )
        if len_invoice_amounts > 1:
            text_input_row[-2].text("=")
            text_input_row[-1].text_input(
                "Amount", f"{st.session_state['invoice_amount_sum']:.2f}"
            )

    # ----------------------
    # H) Subject & Body
    # ----------------------
    default_subject = (
        f"PCard_Simon Group_AMOUNT_{st.session_state['invoice_amount_sum']:.2f}"
    )
    subject_input = st.text_input("Email Subject", value=default_subject)

    default_body = (
        f"Hi {', '.join(selected_to_names)},\n\n"
        f"This is {purchaser_name} from Jon Simon group. Here are the purchase details:\n\n"
        f"Amount: ${st.session_state['invoice_amount_sum']:.2f}\n"
        f"Rationale: {st.session_state['invoice_rationale']}\n"
        f"Account: {grant_code}\n\n"
        f"Best,\n{purchaser_name}\n"
        "---------------------\n"
        f"This is an automated email generated by LLM, please reply to {purchaser_email} for any questions.\n"
    )
    body_input = st.text_area("Email Body", value=default_body, height=240)

    # ----------------------
    # H) Send Email Button
    # ----------------------
    if st.button("Send Email"):
        if not uploaded_invoices:
            st.error("Please upload invoice(s) first.")
        elif (
            st.session_state["invoice_amounts"] == 0
            and not st.session_state["invoice_rationale"]
        ):
            st.warning(
                "Please run 'Process with LLM' first so we have valid invoice data."
            )
        else:
            if not to_emails:
                st.error("No valid 'To' recipients selected.")
            else:
                st.info("Sending email...")
                try:
                    send_email_via_gmail(
                        to_emails=to_emails,
                        cc_emails=cc_emails,
                        subject=subject_input,
                        body=body_input,
                        attachment_files=uploaded_invoices,  # Attach all PDFs
                    )
                    st.success("Email sent successfully!")
                except Exception as e:
                    st.error(f"Error sending email: {e}")


# -----------
# Run App
# -----------
if __name__ == "__main__":
    run_streamlit_app()
