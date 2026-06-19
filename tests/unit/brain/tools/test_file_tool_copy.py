from brain.tools import schemas


def test_read_file_description_discourages_crawl():
    desc = schemas.SCHEMAS["read_file"]["description"].lower()
    assert "do not walk up" in desc or "do not crawl" in desc
    assert "list" in desc and "folder" in desc


def test_list_directory_description_discourages_crawl():
    desc = schemas.SCHEMAS["list_directory"]["description"].lower()
    assert "do not walk up" in desc or "do not crawl" in desc
    assert "list" in desc and "folder" in desc
