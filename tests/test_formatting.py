from interfaces.telegram.formatting import markdown_to_telegram_html, split_message


def test_escapes_html_special_characters():
    assert markdown_to_telegram_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_converts_bold_and_inline_code():
    assert (
        markdown_to_telegram_html("Use **git** and `a<b>`")
        == "Use <b>git</b> and <code>a&lt;b&gt;</code>"
    )


def test_code_block_content_is_escaped_and_not_formatted():
    text = "```python\n**x** = '<y>'\n```"

    assert markdown_to_telegram_html(text) == "<pre>**x** = '&lt;y&gt;'\n</pre>"


def test_heading_becomes_bold_without_nested_tags():
    assert markdown_to_telegram_html("## The **real** fix") == "<b>The real fix</b>"


def test_unmatched_markers_are_left_as_text():
    assert markdown_to_telegram_html("2 ** 3 and a `tick") == "2 ** 3 and a `tick"


def test_short_message_is_not_split():
    assert split_message("hello") == ["hello"]


def test_splits_on_paragraph_boundaries_within_limit():
    paragraphs = ["x" * 40, "y" * 40, "z" * 40]

    chunks = split_message("\n\n".join(paragraphs), limit=90)

    assert chunks == ["x" * 40 + "\n\n" + "y" * 40, "z" * 40]


def test_hard_cuts_text_without_whitespace():
    chunks = split_message("a" * 25, limit=10)

    assert chunks == ["a" * 10, "a" * 10, "a" * 5]
    assert all(len(chunk) <= 10 for chunk in chunks)
