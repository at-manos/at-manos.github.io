---
layout: page
title: About
description: About Me
keywords: Aiden Manos, atmanos
comments: true
menu: about
permalink: /about/
---

Hi, I'm Aiden! I'm a junior at the University of Maryland, College Park, pursuing a Computer Science degree with a minor in Technology Entrepreneurship.

I'm passionate about the intersection of data science, cybersecurity, and systems engineering. My interests span machine learning and neural networks to network security and infrastructure design. I love tackling complex problems that require both technical depth and strategic thinking.

Currently seeking internships where I can apply my curiosity for learning, problem-solving skills, and technical expertise to real-world challenges in data, security, and software development.

When I'm not coding, you'll find me taking photos, studying Chinese, or reading linguistics research.

## Contact

<ul>
{% for website in site.data.social %}
<li>{{website.sitename }}：<a href="{{ website.url }}" target="_blank">@{{ website.name }}</a></li>
{% endfor %}
{% if site.url contains 'blog.atmanos.com' %}
{% endif %}
</ul>

## Skills and Interests

{% for skill in site.data.skills %}
### {{ skill.name }}
<div class="btn-inline">
{% for keyword in skill.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}