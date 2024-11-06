import argparse
import os
import re

import mistune
from slugify import slugify


def convert_markdown_table_to_html(
    markdown,
    html_doc,
    headers=None,
    # center_title= True,
):
    generated_html = mistune.html(markdown)

    # Fix multi-rows
    for num_columns in list(range(10, 1, -1)):
        regex_to = rf'\1<td colspan="{num_columns}" style="text-align: center;"><u>\2</u></td></tr>'
        # if not center_title:
        #     regex_to = rf'\1<td colspan="{num_columns-5}"></td><td colspan="{num_columns-2}"><u>\2</u></td></tr>'
        if headers:
            headers = headers.strip("<>/")

            def regex_to(match):
                title = match.group(2)
                title = re.sub("<[^>]*>", "", title)
                title = f"<{headers} id={slugify(title)}>{title}</{headers}>"
                return f'{match.group(1)}<td colspan="{num_columns}">{title}</td></tr>'

        generated_html = re.sub(
            rf"(<tr>\s*)<td>(.+)</td>\s*(<td>\s*</td>\s*){{{num_columns-1}}}</tr>", regex_to, generated_html
        )

    html_doc.write(generated_html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "README_dataset_table.md"), nargs="?"
    )
    parser.add_argument(
        "output",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "README_dataset_table.html"),
        nargs="?",
    )
    parser.add_argument("--headers", default="h4", help="Use header level instead of just center the category. ex: h4")
    args = parser.parse_args()

    with open(args.input) as f_in:
        table_content = f_in.readlines()

    with open(args.output, "w") as f_out:
        convert_markdown_table_to_html(
            "".join(table_content),
            f_out,
            # headers="h4",
            headers=args.headers,
        )
