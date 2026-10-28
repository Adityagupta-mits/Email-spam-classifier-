/**
 * main.js — Landing page JavaScript
 * Handles smooth navigation, scroll effects, and hero interactions.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── Active nav link on scroll ───────────────────────────
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".landing-nav .nav-links a");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          navLinks.forEach((link) => {
            link.classList.remove("active");
            if (link.getAttribute("href") === `#${entry.target.id}`) {
              link.classList.add("active");
            }
          });
        }
      });
    },
    { threshold: 0.4 }
  );

  sections.forEach((s) => observer.observe(s));

  // ── Navbar shadow on scroll ─────────────────────────────
  const nav = document.querySelector(".landing-nav");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 20) {
      nav.style.boxShadow = "0 2px 16px rgba(0,0,0,.10)";
    } else {
      nav.style.boxShadow = "0 1px 4px rgba(0,0,0,.07)";
    }
  });

  // ── Smooth scroll for anchor links ──────────────────────
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const target = document.querySelector(anchor.getAttribute("href"));
      if (target) {
        e.preventDefault();
        const offset = 70;
        const top = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top, behavior: "smooth" });
      }
    });
  });

  // ── Animate stats counter ───────────────────────────────
  const counters = document.querySelectorAll("[data-count]");
  const countObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          countObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  counters.forEach((c) => countObserver.observe(c));

  function animateCounter(el) {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || "";
    const duration = 1500;
    const step = Math.ceil(target / (duration / 16));
    let current = 0;

    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current.toLocaleString() + suffix;
      if (current >= target) clearInterval(timer);
    }, 16);
  }

  // ── Card entrance animation ─────────────────────────────
  const cards = document.querySelectorAll(".feature-card");
  const cardObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
          setTimeout(() => {
            entry.target.style.opacity = "1";
            entry.target.style.transform = "translateY(0)";
          }, i * 80);
          cardObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1 }
  );

  cards.forEach((card) => {
    card.style.opacity = "0";
    card.style.transform = "translateY(20px)";
    card.style.transition = "opacity 0.5s ease, transform 0.5s ease";
    cardObserver.observe(card);
  });

  // ── Mobile nav toggle ────────────────────────────────────
  const mobileToggle = document.getElementById("mobile-nav-toggle");
  const navMenu = document.getElementById("nav-menu");
  if (mobileToggle && navMenu) {
    mobileToggle.addEventListener("click", () => {
      navMenu.classList.toggle("show");
    });
  }

  // ── Loading Screen Dismiss ───────────────────────────────
  const loader = document.getElementById("loading-screen");
  if (loader) {
    setTimeout(() => {
      loader.classList.add("hidden-loader");
    }, 450);
  }

  // ── Auth Modal Trigger ──────────────────────────────────
  const openLoginBtn = document.getElementById("open-login-btn");
  const closeLoginBtn = document.getElementById("close-modal-btn");
  const authModal = document.getElementById("auth-modal");

  if (openLoginBtn && authModal) {
    openLoginBtn.addEventListener("click", () => authModal.classList.add("show"));
  }
  if (closeLoginBtn && authModal) {
    closeLoginBtn.addEventListener("click", () => authModal.classList.remove("show"));
  }
  if (authModal) {
    authModal.addEventListener("click", (e) => {
      if (e.target === authModal) authModal.classList.remove("show");
    });
  }
});
