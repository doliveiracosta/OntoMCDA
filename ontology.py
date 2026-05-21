"""OWL profile loader for OntoMCDA.

The loader intentionally uses the Python standard library so the public Streamlit
deployment does not depend on heavyweight ontology runtimes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from .constants import ATTRS, ATTR_TO_PROP, DATA_ATTRS
from .text_inference import canon_value

RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"


def local_name(uri: object) -> str:
    text = str(uri)
    if "#" in text:
        return text.split("#")[-1]
    return text.rstrip("/").split("/")[-1]


def tag_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def parse_bool_literal(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"true", "1", "sim", "yes"}:
        return "Sim"
    if text in {"false", "0", "nao", "não", "no"}:
        return "Nao"
    return str(value)


def _first_label(description: ET.Element, fallback: str) -> str:
    for child in description:
        if tag_name(child.tag) in {"label_pt", "label"} and child.text:
            return child.text.strip()
    return fallback


def _values_for_props(description: ET.Element, prop_names: list[str]) -> list[str]:
    values = []
    prop_set = set(prop_names)
    for child in description:
        if tag_name(child.tag) not in prop_set:
            continue
        resource = child.attrib.get(RDF_RESOURCE)
        if resource:
            values.append(local_name(resource))
        elif child.text and child.text.strip():
            values.append(child.text.strip())
    return values


def load_profiles(owl_path: str | Path) -> dict[str, dict[str, Optional[str]]]:
    tree = ET.parse(owl_path)
    root = tree.getroot()
    profiles: dict[str, dict[str, Optional[str]]] = {}

    for description in root.findall("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"):
        about = description.attrib.get(RDF_ABOUT)
        if not about:
            continue

        types = [
            local_name(child.attrib.get(RDF_RESOURCE, ""))
            for child in description
            if tag_name(child.tag) == "type"
        ]
        if "MetodoMCDA" not in types:
            continue

        name = _first_label(description, local_name(about))
        profile: dict[str, Optional[str]] = {"_id": local_name(about), "_label": name, "_classes": ", ".join(types)}

        for attr in ATTRS:
            values = _values_for_props(description, ATTR_TO_PROP[attr])
            if not values:
                profile[attr] = None
            elif attr in DATA_ATTRS:
                profile[attr] = parse_bool_literal(values[0])
            else:
                profile[attr] = canon_value(attr, values[0])

        profiles[name] = profile

    return dict(sorted(profiles.items(), key=lambda item: item[0].lower()))
