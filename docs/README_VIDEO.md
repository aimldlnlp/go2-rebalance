# README video delivery

The README now uses the video attachment reference supplied through the owner's GitHub editor, on its own line. It replaces the temporary local HTML video element and linked-image approach. GitHub should render this form as an inline player rather than a normal text link.

GitHub repository-file URLs and relative MP4 paths do not reliably produce playable README embeds. The working GitHub approach is to upload the MP4 as a video attachment in an editor, then place the returned attachment URL on its own line in the README. GitHub renders the attachment as a player; readers do not need to follow an ordinary link. Uploading an attachment is separate from committing or pushing repository files, but it does send the video to GitHub immediately.

The owner uploaded an attachment in an unsent issue draft and supplied its reference in a screenshot. That exact reference is now in README.md. No issue was submitted. The owner subsequently authorized committing and pushing the repository. An unauthenticated download check still returned HTTP 404 during publication checks, so public accessibility and playback remain unverified. Check the editor's Preview tab in the owner's signed-in session; if it fails, upload the video again and replace the README attachment reference.

Attachment reference: `https://github.com/user-attachments/assets/5098d507-3edc-4f59-9bc2-7a04b6014130`.

The retained file is 51,060,691 bytes. GitHub documents a 10 MB video attachment limit for free plans and 100 MB for paid plans. If the repository uses the free limit, a smaller MP4 attachment will be needed; the retained original should stay unchanged.

References checked on 5 November 2026:

- [GitHub attachment documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)
- [GitHub Community: repository-file links versus video attachments](https://github.com/orgs/community/discussions/19403)
- [GitHub Community: README player and attachment URL handling](https://github.com/orgs/community/discussions/209976)
