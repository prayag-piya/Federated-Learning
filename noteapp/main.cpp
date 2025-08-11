#include <wx/wx.h>
#include <curl/curl.h>
#include <string>
#include <sstream>
#include <iostream>
#include "json.hpp"  // nlohmann::json single header

using json = nlohmann::json;

// Write callback for curl
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

// Sends POST to /predict endpoint, returns predicted next word or empty string on failure
std::string GetNextWord(const std::string& text) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;

    if (!curl) {
        wxLogError("Failed to init CURL");
        return "";
    }

    std::string url = "http://127.0.0.1:8000/predict";

    // Build JSON body
    json body_json = { {"text", text} };
    std::string body = body_json.dump();

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        wxLogError("curl_easy_perform() failed: %s", curl_easy_strerror(res));
        return "";
    }

    try {
        auto response_json = json::parse(readBuffer);
        if (response_json.contains("next_word") && response_json["next_word"].is_string()) {
            return response_json["next_word"].get<std::string>();
        }
    } catch (json::parse_error& e) {
        wxLogError("JSON parse error: %s", e.what());
    } catch (std::exception& e) {
        wxLogError("Exception: %s", e.what());
    }

    return "";
}

class MyFrame : public wxFrame {
public:
    wxTextCtrl* textCtrl;
    wxListBox* suggestions;

    MyFrame() : wxFrame(nullptr, wxID_ANY, "Smart Writer", wxDefaultPosition, wxSize(600, 400)) {
        wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

        textCtrl = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition, wxSize(600, 300), wxTE_MULTILINE);
        suggestions = new wxListBox(this, wxID_ANY);

        sizer->Add(textCtrl, 1, wxEXPAND | wxALL, 5);
        sizer->Add(suggestions, 0, wxEXPAND | wxALL, 5);

        SetSizer(sizer);

        textCtrl->Bind(wxEVT_TEXT, &MyFrame::OnTextChanged, this);
    }

    void OnTextChanged(wxCommandEvent& event) {
        wxString content = textCtrl->GetValue();
        wxArrayString words = wxSplit(content, ' ', '\0');  // split by space

        if (!words.IsEmpty()) {
            wxString lastWord = words.Last();
            if (!lastWord.IsEmpty()) {
                // Call FastAPI with the last word (blocking call; for production use threading)
                std::string predicted = GetNextWord(std::string(lastWord.mb_str()));
                suggestions->Clear();
                if (!predicted.empty()) {
                    suggestions->Append(predicted);
                }
            }
        } else {
            suggestions->Clear();
        }

        event.Skip();
    }
};

class MyApp : public wxApp {
public:
    bool OnInit() override {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) {
            wxMessageBox("Failed to initialize curl", "Error", wxICON_ERROR);
            return false;
        }

        MyFrame* frame = new MyFrame();
        frame->Show();

        return true;
    }

    int OnExit() override {
        curl_global_cleanup();
        return 0;
    }
};

wxIMPLEMENT_APP(MyApp);
