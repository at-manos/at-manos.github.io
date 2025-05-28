---
layout: page
title: Links
description: Links I find interesting or useful
keywords: links
comments: true
menu: links
permalink: /links/
---
> Random links I find interesting or useful.
<ul>
{% for link in site.data.links %}
  {% if link.src == 'www' %}
  <li><a href="{{ link.url }}" target="_blank">{{ link.name}}</a></li>
  {% endif %}
{% endfor %}
</ul>
