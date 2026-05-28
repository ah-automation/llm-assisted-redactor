import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAGE_WIDTH = 1800
PAGE_HEIGHT = 2400
MARGIN = 120


CUSTOMERS = [
    {
        "name": "Maya Chen",
        "company": "Northstar Gardens",
        "address": ["1847 Maple Ridge Road", "Burlington, ON L7M 4A5"],
        "email": "maya.chen@example.test",
        "phone": "416-555-0184",
        "account": "ACCT-904182",
    },
    {
        "name": "Jordan Patel",
        "company": "Patel Family Holdings",
        "address": ["92 Cedar Lane", "Austin, TX 78704"],
        "email": "jordan.patel@example.test",
        "phone": "512-555-0172",
        "account": "CUST-772019",
    },
    {
        "name": "Rosa Martin",
        "company": "R. Martin Studio",
        "address": ["501 West 22nd Street, Apt 8B", "New York, NY 10011"],
        "email": "rosa.martin@example.test",
        "phone": "212-555-0149",
        "account": "CLIENT-3107",
    },
    {
        "name": "Elliot Walker",
        "company": "Walker Home Services",
        "address": ["77 Pine Creek Drive", "Calgary, AB T2P 3H9"],
        "email": "elliot.walker@example.test",
        "phone": "403-555-0198",
        "account": "REF-600245",
    },
]


VENDORS = [
    ("Blue Harbor Consulting", "INVOICE", (28, 78, 132)),
    ("Pine & Stone Utilities", "STATEMENT", (37, 105, 83)),
    ("Ardent Medical Supply", "SERVICE INVOICE", (127, 55, 78)),
    ("Civic Field Contractors", "TAX INVOICE", (92, 83, 64)),
    ("ParcelPoint Retail", "ORDER INVOICE", (80, 76, 150)),
]


ITEMS = [
    ("Consulting services", 6, 145.00),
    ("Equipment rental", 2, 315.00),
    ("Monthly service plan", 1, 89.95),
    ("Installation labour", 4, 112.50),
    ("Replacement parts", 3, 48.25),
    ("Delivery fee", 1, 24.99),
    ("Administrative fee", 1, 18.00),
]


def get_font(size, bold=False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text(draw, xy, text, size=34, fill=(20, 24, 28), bold=False):
    draw.text(xy, text, font=get_font(size, bold), fill=fill)


def draw_lines(draw, x, y, lines, size=34, gap=44, fill=(20, 24, 28), bold=False):
    for line in lines:
        draw_text(draw, (x, y), line, size=size, fill=fill, bold=bold)
        y += gap
    return y


def draw_rule(draw, y, fill=(210, 214, 220)):
    draw.line([(MARGIN, y), (PAGE_WIDTH - MARGIN, y)], fill=fill, width=3)


def money(value):
    return f"${value:,.2f}"


def draw_table(draw, x, y, accent, compact=False):
    selected_items = random.sample(ITEMS, 4)
    row_height = 64 if compact else 72
    columns = [x, x + 900, x + 1080, x + 1290]
    headers = ["Description", "Qty", "Rate", "Amount"]

    draw.rounded_rectangle(
        [x, y, PAGE_WIDTH - MARGIN, y + row_height],
        radius=0,
        fill=accent,
    )
    for index, header in enumerate(headers):
        draw_text(draw, (columns[index] + 12, y + 16), header, size=28, fill="white", bold=True)

    y += row_height
    total = 0
    for item, quantity, rate in selected_items:
        amount = quantity * rate
        total += amount
        draw.rectangle([x, y, PAGE_WIDTH - MARGIN, y + row_height], outline=(220, 224, 230), width=2)
        draw_text(draw, (columns[0] + 12, y + 18), item, size=28)
        draw_text(draw, (columns[1] + 12, y + 18), str(quantity), size=28)
        draw_text(draw, (columns[2] + 12, y + 18), money(rate), size=28)
        draw_text(draw, (columns[3] + 12, y + 18), money(amount), size=28)
        y += row_height

    tax = total * 0.13
    grand_total = total + tax
    y += 34
    summary_x = PAGE_WIDTH - 540
    for label, value in [("Subtotal", total), ("Tax", tax), ("Total Due", grand_total)]:
        draw_text(draw, (summary_x, y), label, size=30, bold=label == "Total Due")
        draw_text(draw, (PAGE_WIDTH - 290, y), money(value), size=30, bold=label == "Total Due")
        y += 48


def draw_footer(draw, invoice_number):
    draw_rule(draw, PAGE_HEIGHT - 220)
    draw_text(draw, (MARGIN, PAGE_HEIGHT - 180), "Payment reference", size=28, fill=(90, 96, 106))
    draw_text(draw, (MARGIN, PAGE_HEIGHT - 140), f"PAY-{invoice_number}", size=34, bold=True)
    draw_text(
        draw,
        (PAGE_WIDTH - 670, PAGE_HEIGHT - 150),
        "This synthetic invoice is for local redaction testing only.",
        size=24,
        fill=(110, 116, 126),
    )


def layout_classic(draw, customer, vendor, title, accent, invoice_number):
    draw_text(draw, (MARGIN, 100), vendor, size=54, bold=True, fill=accent)
    draw_text(draw, (PAGE_WIDTH - 520, 100), title, size=62, bold=True)
    draw_text(draw, (PAGE_WIDTH - 520, 190), f"Invoice No: {invoice_number}", size=32)
    draw_text(draw, (PAGE_WIDTH - 520, 236), "Issue Date: 2026-05-27", size=32)
    draw_text(draw, (PAGE_WIDTH - 520, 282), "Due Date: 2026-06-26", size=32)
    draw_rule(draw, 370)

    draw_text(draw, (MARGIN, 430), "Bill To", size=34, bold=True, fill=accent)
    draw_lines(
        draw,
        MARGIN,
        486,
        [customer["name"], customer["company"], *customer["address"], customer["email"], customer["phone"]],
        size=34,
    )

    draw_text(draw, (1050, 430), "Account", size=34, bold=True, fill=accent)
    draw_lines(draw, 1050, 486, [customer["account"], "Terms: Net 30"], size=34)
    draw_table(draw, MARGIN, 850, accent)


def layout_service(draw, customer, vendor, title, accent, invoice_number):
    draw.rectangle([0, 0, PAGE_WIDTH, 310], fill=accent)
    draw_text(draw, (MARGIN, 80), vendor, size=52, bold=True, fill="white")
    draw_text(draw, (MARGIN, 170), title, size=42, fill="white")
    draw_text(draw, (PAGE_WIDTH - 540, 80), f"Invoice: {invoice_number}", size=34, fill="white", bold=True)
    draw_text(draw, (PAGE_WIDTH - 540, 132), "Service Date: 2026-05-12", size=30, fill="white")

    draw_text(draw, (MARGIN, 390), "Customer Information", size=36, bold=True, fill=accent)
    draw_lines(draw, MARGIN, 450, [customer["name"], *customer["address"]], size=34)
    draw_text(draw, (MARGIN, 625), f"Phone: {customer['phone']}", size=34)
    draw_text(draw, (MARGIN, 675), f"Email: {customer['email']}", size=34)

    draw_text(draw, (1050, 390), "Customer No.", size=36, bold=True, fill=accent)
    draw_text(draw, (1050, 450), customer["account"], size=38, bold=True)
    draw_text(draw, (1050, 520), "Work Order", size=34, bold=True, fill=accent)
    draw_text(draw, (1050, 580), f"WO-{random.randint(21000, 98999)}", size=38, bold=True)
    draw_table(draw, MARGIN, 820, accent, compact=True)


def layout_shipping(draw, customer, vendor, title, accent, invoice_number):
    draw_text(draw, (MARGIN, 90), title, size=64, bold=True, fill=accent)
    draw_text(draw, (MARGIN, 170), vendor, size=36)
    draw_text(draw, (MARGIN, 240), f"Invoice Number {invoice_number}", size=34)
    draw_text(draw, (MARGIN, 286), "Invoice Date 2026-05-27", size=34)

    draw.rectangle([MARGIN, 390, PAGE_WIDTH - MARGIN, 760], outline=(205, 210, 218), width=3)
    draw.line([(PAGE_WIDTH // 2, 390), (PAGE_WIDTH // 2, 760)], fill=(205, 210, 218), width=3)
    draw_text(draw, (MARGIN + 34, 430), "Bill To", size=34, bold=True, fill=accent)
    draw_lines(draw, MARGIN + 34, 486, [customer["name"], *customer["address"], customer["email"]], size=32, gap=42)
    draw_text(draw, (PAGE_WIDTH // 2 + 34, 430), "Ship To", size=34, bold=True, fill=accent)
    draw_lines(draw, PAGE_WIDTH // 2 + 34, 486, [customer["company"], *customer["address"]], size=32, gap=42)

    draw_text(draw, (MARGIN, 820), f"Customer ID: {customer['account']}", size=34, bold=True)
    draw_text(draw, (MARGIN, 870), f"Contact: {customer['phone']}", size=34)
    draw_table(draw, MARGIN, 980, accent)


LAYOUTS = [layout_classic, layout_service, layout_shipping]


def generate_invoice(index, output_dir):
    random.seed(index)
    customer = CUSTOMERS[index % len(CUSTOMERS)]
    vendor, title, accent = VENDORS[index % len(VENDORS)]
    invoice_number = f"INV-{202600 + index:06d}"

    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    LAYOUTS[index % len(LAYOUTS)](draw, customer, vendor, title, accent, invoice_number)
    draw_footer(draw, invoice_number)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"invoice_{index + 1:02d}.png"
    image.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic invoice images for local redaction testing.")
    parser.add_argument("--count", type=int, default=6, help="Number of invoice images to generate.")
    parser.add_argument("--output-dir", default="input/Invoices", help="Directory for generated invoice images.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    paths = [generate_invoice(index, output_dir) for index in range(args.count)]
    for path in paths:
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
