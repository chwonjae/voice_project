import streamlit as st
import time
from typing import Optional

from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="SUL Bank Demo",
    page_icon="💶",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Demo Constants
# ============================================================
SAMPLE_SCREEN_SPLASH = "splash"
SAMPLE_SCREEN_HOME = "home"
SAMPLE_SCREEN_RECIPIENT = "recipient"
SAMPLE_SCREEN_AMOUNT = "amount"

SAMPLE_AVAILABLE_BANK_OPTIONS = [
    "신한은행",
    "국민은행",
    "우리은행",
    "하나은행",
    "농협은행",
    "카카오뱅크",
    "토스뱅크",
    "기업은행",
]


# ============================================================
# Global Style
# ============================================================
def inject_sample_global_styles() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Pretendard", "Noto Sans KR", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top, rgba(31,113,255,0.06), transparent 18%),
                linear-gradient(180deg, #f4f6fa 0%, #eef2f7 100%);
        }

        section.main > div {
            max-width: 460px;
            padding-top: 0.75rem;
            padding-bottom: 1.2rem;
        }

        div[data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f5f7fb 0%, #eff3f8 100%);
        }

        div[data-testid="stHeader"] {
            background: transparent;
        }

        div[data-testid="collapsedControl"] {
            display: none;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .sample-screen-fade {
            animation: sampleFadeUp 0.45s ease-out;
        }

        @keyframes sampleFadeUp {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0px);
            }
        }

        .sample-brand-text {
            letter-spacing: -0.02em;
            letter-spacing: -0.02em;
            color: #1452d9;
        }

        .sample-muted-caption {
            color: #7b8597;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .sample-section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #162033;
            margin-bottom: 10px;
        }

        .sample-bank-card-title {
            color: rgba(255,255,255,0.92);
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .sample-bank-card-account {
            color: rgba(255,255,255,0.82);
            font-size: 0.82rem;
            margin-bottom: 12px;
        }

        .sample-bank-card-balance {
            color: white;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .sample-card-label {
            font-size: 0.76rem;
            color: #7b8597;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .sample-card-title {
            font-size: 1rem;
            color: #111827;
            font-weight: 700;
            margin-top: 2px;
        }

        .sample-card-subtitle {
            font-size: 0.84rem;
            color: #6b7280;
            margin-top: 5px;
            line-height: 1.45;
        }

        .sample-home-icon-badge {
            width: 36px;
            height: 36px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f2f6ff;
            font-size: 1rem;
        }

        .sample-hero-question {
            font-size: 1.85rem;
            line-height: 1.18;
            font-weight: 800;
            color: #111827;
            letter-spacing: -0.03em;
            margin-bottom: 4px;
        }

        .sample-amount-value {
            text-align: center;
            font-size: 2.25rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.04em;
            margin-top: 12px;
            margin-bottom: 8px;
        }

        .sample-amount-help {
            text-align: center;
            font-size: 0.88rem;
            color: #6b7280;
        }

        .sample-summary-chip {
            padding: 10px 12px;
            background: #f3f6fb;
            border: 1px solid #e5ebf4;
            border-radius: 14px;
            font-size: 0.82rem;
            color: #475569;
            margin-bottom: 8px;
        }

        .sample-bottom-nav-wrap {
            margin-top: 18px;
            padding: 8px 6px 0 6px;
            border-top: 1px solid rgba(15, 23, 42, 0.06);
        }

        .sample-bottom-nav-item {
            text-align: center;
            padding: 6px 0;
        }

        .sample-bottom-nav-icon {
            font-size: 1.06rem;
            line-height: 1.1;
            margin-bottom: 4px;
        }

        .sample-bottom-nav-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: #8a94a6;
        }

        .sample-bottom-nav-label-active,
        .sample-bottom-nav-icon-active {
            color: #1d4ed8;
        }

        .sample-splash-wrap {
            min-height: 770px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            text-align: center;
        }

        .sample-splash-logo {
            width: 92px;
            height: 92px;
            border-radius: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.4rem;
            background: linear-gradient(145deg, #1d4ed8 0%, #2563eb 55%, #60a5fa 100%);
            color: white;
            box-shadow: 0 18px 40px rgba(37, 99, 235, 0.28);
            margin-bottom: 18px;
        }

        .sample-splash-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.04em;
            margin-bottom: 8px;
        }

        .sample-splash-subtitle {
            font-size: 0.95rem;
            line-height: 1.55;
            color: #64748b;
            max-width: 270px;
            margin: 0 auto 18px auto;
        }

        .sample-inline-note {
            margin-top: 10px;
            font-size: 0.78rem;
            color: #94a3b8;
            text-align: center;
        }

        div[data-testid="stButton"] > button {
            border-radius: 14px;
            min-height: 48px;
            font-weight: 700;
            border: 1px solid rgba(15, 23, 42, 0.07);
            transition: all 0.18s ease;
            box-shadow: none;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            border-color: rgba(37, 99, 235, 0.18);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: 14px !important;
            min-height: 52px !important;
            border-color: #dbe3ef !important;
            background: #fbfcff !important;
        }

        .sample-hidden-label {
            margin-top: -8px;
        }

        @media (max-width: 520px) {
            section.main > div {
                padding-left: 0.4rem;
                padding-right: 0.4rem;
            }

            .sample-splash-wrap {
                min-height: 86vh;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# State Initialization
# ============================================================
def initialize_sample_mock_bank_data() -> None:
    if "sample_fake_account_name" not in st.session_state:
        st.session_state.sample_fake_account_name = "SUL 주거래 우대통장"
    if "sample_fake_account_number" not in st.session_state:
        st.session_state.sample_fake_account_number = "110-482-190284"
    if "sample_fake_account_balance" not in st.session_state:
        st.session_state.sample_fake_account_balance = 12850320
    if "sample_fake_card_name" not in st.session_state:
        st.session_state.sample_fake_card_name = "SUL Prime 카드"
    if "sample_fake_card_masked_number" not in st.session_state:
        st.session_state.sample_fake_card_masked_number = "1234 •••• •••• 9402"


def initialize_sample_app_state() -> None:
    initialize_sample_mock_bank_data()

    defaults = {
        "sample_current_screen": SAMPLE_SCREEN_SPLASH,
        "sample_selected_bank_name": SAMPLE_AVAILABLE_BANK_OPTIONS[0],
        "sample_recipient_account_number": "",
        "sample_transfer_amount": 0,
        "sample_transfer_amount_display": "0원",
        "sample_home_tab": "home",
        "sample_is_demo_mode": True,
        "sample_recent_action_message": "",
        "sample_has_initialized_demo_data": True,
        "sample_should_auto_advance_from_splash": True,
        "sample_splash_has_advanced": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Formatters
# ============================================================
def format_sample_currency_krw(amount: int) -> str:
    return f"{amount:,}원"


def format_sample_amount_for_display() -> None:
    amount = int(st.session_state.sample_transfer_amount)
    st.session_state.sample_transfer_amount_display = format_sample_currency_krw(amount)


# ============================================================
# Navigation
# ============================================================
def sample_navigate_to_screen(target_screen: str) -> None:
    st.session_state.sample_current_screen = target_screen
    st.rerun()


def sample_go_home() -> None:
    st.session_state.sample_current_screen = SAMPLE_SCREEN_HOME
    st.rerun()


def sample_go_back_from_current_screen() -> None:
    current_screen = st.session_state.sample_current_screen
    if current_screen == SAMPLE_SCREEN_RECIPIENT:
        st.session_state.sample_current_screen = SAMPLE_SCREEN_HOME
    elif current_screen == SAMPLE_SCREEN_AMOUNT:
        st.session_state.sample_current_screen = SAMPLE_SCREEN_RECIPIENT
    else:
        st.session_state.sample_current_screen = SAMPLE_SCREEN_HOME
    st.rerun()


# ============================================================
# Sample Mock Functions (Future FastAPI Replacement Points)
# ============================================================
def sample_fetch_mock_account_summary() -> dict:
    return {
        "account_name": st.session_state.sample_fake_account_name,
        "account_number": st.session_state.sample_fake_account_number,
        "account_balance": st.session_state.sample_fake_account_balance,
    }


def sample_validate_mock_recipient_account(selected_bank_name: str, recipient_account_number: str) -> bool:
    if not selected_bank_name:
        return False
    sanitized = recipient_account_number.replace("-", "").replace(" ", "")
    return sanitized.isdigit() and len(sanitized) >= 8


def sample_submit_mock_transfer_request() -> dict:
    return {
        "success": True,
        "message": "데모용 송금 요청이 생성되었습니다. 실제 연동 시 FastAPI 요청으로 교체하세요.",
        "recipient_bank": st.session_state.sample_selected_bank_name,
        "recipient_account": st.session_state.sample_recipient_account_number,
        "amount": st.session_state.sample_transfer_amount,
    }


# ============================================================
# Components
# ============================================================
def render_sample_top_app_bar(
    left_button_label: Optional[str] = None,
    center_title: str = "",
    right_button_label: Optional[str] = None,
    left_action=None,
    right_action=None,
) -> None:
    left_col, center_col, right_col = st.columns([1, 4, 1])

    with left_col:
        if left_button_label:
            if st.button(left_button_label, key=f"top_left_{center_title}_{left_button_label}", use_container_width=True):
                if left_action:
                    left_action()
        else:
            st.write("")

    with center_col:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:10px; font-size:1rem; font-weight:800; color:#0f172a;">
                {center_title}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        if right_button_label:
            if st.button(right_button_label, key=f"top_right_{center_title}_{right_button_label}", use_container_width=True):
                if right_action:
                    right_action()
        else:
            st.write("")


def render_sample_home_header() -> None:
    left_col, center_col, right_col = st.columns([3.5, 1, 1])
    with left_col:
        st.markdown('<div><h2 class="sample-brand-text">💶 SUL Bank</h2></div>', unsafe_allow_html=True)
    with center_col:
        st.button("🔍", key="sample_home_search_button", use_container_width=True, disabled=True)
    with right_col:
        st.button("🔔", key="sample_home_alert_button", use_container_width=True, disabled=True)


def render_sample_section_header(title: str) -> None:
    st.markdown(f'<div class="sample-section-title">{title} &gt;</div>', unsafe_allow_html=True)


def render_sample_account_transfer_card() -> None:
    account_summary = sample_fetch_mock_account_summary()

    with stylable_container(
        key="sample_transfer_account_card_container",
        css_styles="""
            {
                background: linear-gradient(145deg, #1452d9 0%, #2563eb 58%, #4f8df7 100%);
                border-radius: 22px;
                padding: 18px 18px 18px 18px;
                box-shadow: 0 16px 34px rgba(37, 99, 235, 0.24);
                border: 1px solid rgba(255,255,255,0.12);
            }
            button {
                background: rgba(255,255,255,0.18) !important;
                color: white !important;
                border: 1px solid rgba(255,255,255,0.20) !important;
                border-radius: 14px !important;
                min-height: 46px !important;
            }
        """,
    ):
        top_left, top_right = st.columns([4, 1])
        with top_left:
            st.markdown(
                f"""
                <div class="sample-bank-card-title">💶 {account_summary['account_name']}</div>
                <div class="sample-bank-card-account">{account_summary['account_number']}</div>
                <div class="sample-bank-card-balance">{format_sample_currency_krw(account_summary['account_balance'])}</div>
                """,
                unsafe_allow_html=True,
            )
        with top_right:
            st.markdown(
                """
                <div style="display:flex; justify-content:flex-end; align-items:flex-start; height:100%;">
                    <div style="width:42px; height:42px; border-radius:14px; background:rgba(255,255,255,0.16); display:flex; align-items:center; justify-content:center; color:white; font-size:1.1rem; font-weight:700;">⇄</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        add_vertical_space(1)
        if st.button("이체", key="sample_go_transfer_from_account_card", use_container_width=True):
            sample_navigate_to_screen(SAMPLE_SCREEN_RECIPIENT)


def render_sample_mock_info_card(
    container_key: str,
    emoji: str,
    label: str,
    title: str,
    subtitle: str,
) -> None:
    with stylable_container(
        key=container_key,
        css_styles="""
            {
                background: rgba(255,255,255,0.94);
                border: 1px solid rgba(15, 23, 42, 0.05);
                border-radius: 20px;
                padding: 16px;
                padding-bottom: 48px;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        icon_col, text_col = st.columns([1, 4])
        with icon_col:
            st.markdown(f'<div class="sample-home-icon-badge">{emoji}</div>', unsafe_allow_html=True)
        with text_col:
            st.markdown(
                f"""
                <div class="sample-card-label">{label}</div>
                <div class="sample-card-title">{title}</div>
                <div class="sample-card-subtitle">{subtitle}</div>
                """,
                unsafe_allow_html=True,
            )


def render_sample_bottom_navigation() -> None:
    with stylable_container(
        key="sample_bottom_navigation_container",
        css_styles="""
            {
                background: rgba(255,255,255,0.72);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(15, 23, 42, 0.05);
                border-radius: 20px;
                padding: 8px 8px 6px 8px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            }
        """,
    ):
        st.markdown('<div class="sample-bottom-nav-wrap">', unsafe_allow_html=True)
        cols = st.columns(5)
        items = [
            ("🏠", "홈", True),
            ("💳", "금융", False),
            ("🎁", "혜택", False),
            ("📈", "주식", False),
            ("☰", "전체", False),
        ]
        for col, (icon, label, is_active) in zip(cols, items):
            with col:
                icon_class = "sample-bottom-nav-icon sample-bottom-nav-icon-active" if is_active else "sample-bottom-nav-icon"
                label_class = "sample-bottom-nav-label sample-bottom-nav-label-active" if is_active else "sample-bottom-nav-label"
                st.markdown(
                    f"""
                    <div class="sample-bottom-nav-item">
                        <div class="{icon_class}">{icon}</div>
                        <div class="{label_class}">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Screen Renderers
# ============================================================
def render_sample_splash_screen() -> None:
    st.markdown(
        """
        <div class="sample-splash-wrap">
            <div class="sample-splash-logo">💶</div>
            <div class="sample-splash-title">SUL Bank</div>
            <div class="sample-splash-subtitle">
                Streamlit로 구현한 모바일 뱅킹 시연 프로토타입<br/>
                secure, simple, seamless.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.sample_should_auto_advance_from_splash and not st.session_state.sample_splash_has_advanced:
        st.session_state.sample_splash_has_advanced = True
        time.sleep(1.2)
        sample_navigate_to_screen(SAMPLE_SCREEN_HOME)

    if st.button("데모 시작하기", key="sample_splash_start_button", use_container_width=True):
        sample_navigate_to_screen(SAMPLE_SCREEN_HOME)

    st.markdown('<div class="sample-inline-note">Demo UI • sample flow only</div>', unsafe_allow_html=True)


def render_sample_home_screen() -> None:
    render_sample_home_header()
    add_vertical_space(1)

    colored_header(
        label="내 금융 홈",
        description="실제 은행 앱처럼 보이는 송금 시연용 목업 화면입니다.",
        color_name="blue-70",
    )

    render_sample_section_header("은행")
    render_sample_account_transfer_card()

    add_vertical_space(1)
    render_sample_section_header("카드")
    render_sample_mock_info_card(
        container_key="sample_mock_card_section",
        emoji="💳",
        label="CARD",
        title=f"{st.session_state.sample_fake_card_name}",
        subtitle=f"{st.session_state.sample_fake_card_masked_number} · 이번 달 결제예정 428,500원",
    )

    add_vertical_space(1)
    render_sample_section_header("증권")
    render_sample_mock_info_card(
        container_key="sample_mock_stock_section",
        emoji="📈",
        label="INVEST",
        title="SUL 투자계좌",
        subtitle="해외주식 · ETF · 모의 수익률 +4.28%",
    )

    add_vertical_space(1)
    render_sample_section_header("보험")
    render_sample_mock_info_card(
        container_key="sample_mock_insurance_section",
        emoji="🛡️",
        label="INSURANCE",
        title="생활안심 보험 관리",
        subtitle="보장 확인, 납입 일정, 증명서 발급을 한 번에 확인하세요.",
    )

    render_sample_bottom_navigation()


def handle_sample_recipient_form_submission() -> None:
    is_valid = sample_validate_mock_recipient_account(
        selected_bank_name=st.session_state.sample_selected_bank_name,
        recipient_account_number=st.session_state.sample_recipient_account_number,
    )

    if not is_valid:
        st.session_state.sample_recent_action_message = "기관과 계좌번호를 정확히 입력해 주세요."
        return

    st.session_state.sample_recent_action_message = ""
    sample_navigate_to_screen(SAMPLE_SCREEN_AMOUNT)



def render_sample_recipient_screen() -> None:
    render_sample_top_app_bar(
        left_button_label="←",
        center_title="받는 분 선택",
        right_button_label="⌂",
        left_action=sample_go_back_from_current_screen,
        right_action=sample_go_home,
    )

    add_vertical_space(1)
    st.markdown('<div class="sample-hero-question">누구에게<br/>보낼까요?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-muted-caption">은행과 계좌번호를 입력하면 다음 단계로 이동합니다.</div>', unsafe_allow_html=True)
    add_vertical_space(1)

    with stylable_container(
        key="sample_recipient_form_container",
        css_styles="""
            {
                background: rgba(255,255,255,0.95);
                border-radius: 22px;
                padding: 16px;
                border: 1px solid rgba(15, 23, 42, 0.05);
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        st.selectbox(
            "기관 선택",
            options=SAMPLE_AVAILABLE_BANK_OPTIONS,
            key="sample_selected_bank_name",
            help="데모용 선택 필드입니다. 추후 실제 기관 조회 API와 연결할 수 있습니다.",
        )
        add_vertical_space(1)
        st.text_input(
            "계좌번호 입력",
            key="sample_recipient_account_number",
            placeholder="'-' 없이 계좌번호를 입력하세요",
            help="데모용 계좌번호 입력 필드입니다.",
        )

    if st.session_state.sample_recent_action_message:
        st.warning(st.session_state.sample_recent_action_message)

    add_vertical_space(2)
    if st.button("확인", key="sample_recipient_confirm_button", use_container_width=True):
        handle_sample_recipient_form_submission()


# ============================================================
# Amount Keypad Logic
# ============================================================
def handle_sample_amount_keypad_input(key: str) -> None:
    current_amount = str(st.session_state.sample_transfer_amount)

    if current_amount == "0":
        current_amount = ""

    next_amount_str = current_amount + key
    next_amount_str = next_amount_str.lstrip("0") or "0"

    try:
        st.session_state.sample_transfer_amount = int(next_amount_str)
    except ValueError:
        st.session_state.sample_transfer_amount = 0

    format_sample_amount_for_display()



def handle_sample_amount_delete() -> None:
    current_amount = str(st.session_state.sample_transfer_amount)
    next_amount_str = current_amount[:-1]
    if next_amount_str == "":
        next_amount_str = "0"
    st.session_state.sample_transfer_amount = int(next_amount_str)
    format_sample_amount_for_display()



def handle_sample_transfer_submit() -> None:
    if st.session_state.sample_transfer_amount <= 0:
        st.session_state.sample_recent_action_message = "보낼 금액을 입력해 주세요."
        return

    response = sample_submit_mock_transfer_request()
    st.session_state.sample_recent_action_message = response["message"]



def render_sample_keypad_button(label: str, button_key: str, on_click_action=None, disabled: bool = False) -> None:
    if st.button(label, key=button_key, use_container_width=True, disabled=disabled):
        if on_click_action:
            on_click_action()
        st.rerun()



def render_sample_amount_screen() -> None:
    render_sample_top_app_bar(
        left_button_label="←",
        center_title="보낼 금액 입력",
        right_button_label="⌂",
        left_action=sample_go_back_from_current_screen,
        right_action=sample_go_home,
    )

    add_vertical_space(1)
    st.markdown('<div class="sample-hero-question">얼마를<br/>보낼까요?</div>', unsafe_allow_html=True)

    add_vertical_space(1)
    with stylable_container(
        key="sample_amount_summary_container",
        css_styles="""
            {
                background: rgba(255,255,255,0.95);
                border-radius: 22px;
                padding: 16px;
                border: 1px solid rgba(15, 23, 42, 0.05);
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            }
        """,
    ):
        st.markdown(
            f'<div class="sample-summary-chip">받는 은행 · {st.session_state.sample_selected_bank_name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sample-summary-chip">계좌번호 · {st.session_state.sample_recipient_account_number}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sample-amount-value">{st.session_state.sample_transfer_amount_display}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sample-amount-help">숫자 키패드로 금액을 입력해 주세요.</div>', unsafe_allow_html=True)

    if st.session_state.sample_recent_action_message:
        st.info(st.session_state.sample_recent_action_message)

    add_vertical_space(1)
    with stylable_container(
        key="sample_transfer_submit_container",
        css_styles="""
            {
                background: transparent;
            }
            button {
                background: linear-gradient(145deg, #1452d9 0%, #2563eb 100%) !important;
                color: white !important;
                border: none !important;
                min-height: 52px !important;
                font-weight: 800 !important;
                box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
            }
        """,
    ):
        if st.button("전송", key="sample_amount_transfer_button", use_container_width=True):
            handle_sample_transfer_submit()
            st.rerun()

    add_vertical_space(1)
    keypad_rows = [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7", "8", "9"],
        ["00", "0", "←삭제"],
    ]

    for row_index, row_values in enumerate(keypad_rows):
        cols = st.columns(3)
        for col, key_value in zip(cols, row_values):
            with col:
                with stylable_container(
                    key=f"sample_keypad_container_{row_index}_{key_value}",
                    css_styles="""
                        {
                            background: transparent;
                        }
                        button {
                            background: rgba(255,255,255,0.96) !important;
                            color: #0f172a !important;
                            border: 1px solid rgba(15, 23, 42, 0.06) !important;
                            border-radius: 18px !important;
                            min-height: 66px !important;
                            font-size: 1.05rem !important;
                            font-weight: 800 !important;
                            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
                        }
                    """,
                ):
                    if key_value == "←삭제":
                        render_sample_keypad_button(
                            label="⌫",
                            button_key=f"sample_keypad_delete_{row_index}",
                            on_click_action=handle_sample_amount_delete,
                        )
                    else:
                        render_sample_keypad_button(
                            label=key_value,
                            button_key=f"sample_keypad_{row_index}_{key_value}",
                            on_click_action=lambda value=key_value: handle_sample_amount_keypad_input(value),
                        )

# ============================================================
# Router
# ============================================================
def render_sample_current_screen() -> None:
    current_screen = st.session_state.sample_current_screen

    if current_screen == SAMPLE_SCREEN_SPLASH:
        render_sample_splash_screen()
    elif current_screen == SAMPLE_SCREEN_HOME:
        render_sample_home_screen()
    elif current_screen == SAMPLE_SCREEN_RECIPIENT:
        render_sample_recipient_screen()
    elif current_screen == SAMPLE_SCREEN_AMOUNT:
        render_sample_amount_screen()
    else:
        render_sample_home_screen()


# ============================================================
# App Entrypoint
# ============================================================
def main() -> None:
    inject_sample_global_styles()
    initialize_sample_app_state()
    format_sample_amount_for_display()
    render_sample_current_screen()


if __name__ == "__main__":
    main()
